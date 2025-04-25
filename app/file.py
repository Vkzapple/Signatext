import os
import sys
import time
import pathlib

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import requests
import logging
import torch
import cloudinary
import cloudinary.uploader
import cloudinary.api

# Fix for Windows loading models saved with PosixPath
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

# Force CPU usage
os.environ["YOLO_DEVICE"] = "cpu"

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,  # Debug level for verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('app.file')

# Flask app
tools = Flask(__name__)
app = Flask(__name__)
CORS(app)

# Cloudinary Configuration
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME', 'dljflfis5'),
    api_key=os.getenv('CLOUDINARY_API_KEY', '688273164556944'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET', 'qiak8-mntievgXgTH6XUPg5b0S0')
)
logger.info("Cloudinary configured successfully")

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "model", "bisindo_best.pt")
YOLOV5_DIR = os.path.join(ROOT_DIR, "yolov5")

# Validate model file
if not os.path.isfile(MODEL_PATH):
    logger.error(f"❌ Model not found at {MODEL_PATH}")
    sys.exit(1)

DEVICE = torch.device("cpu")

# Add YOLOv5 repo to path
sys.path.insert(0, YOLOV5_DIR)
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.augmentations import letterbox

# Custom implementation of scale_coords and clip_coords
def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

# Custom non_max_suppression function
def custom_non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, max_det=100):
    """
    Modified version of non_max_suppression that uses class confidence instead of objectness × class confidence
    Works with both tensor and list inputs
    """
    from utils.general import xywh2xyxy, box_iou
    
    # Handle if prediction is a list (which appears to be the case in your error)
    if isinstance(prediction, list):
        return [custom_non_max_suppression(p, conf_thres, iou_thres, classes, max_det)[0] 
                for p in prediction]
    
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > 0.001  # candidates with minimal objectness
    
    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        
        # If none remain process next image
        if not x.shape[0]:
            continue
            
        # Compute class confidence instead of combined confidence
        class_conf, class_pred = x[:, 5:].max(1, keepdim=True)  # only use class confidence for threshold
        x = torch.cat((x[:, :4], class_conf, class_pred.float()), 1)
        
        # Apply conf_thres to class confidence instead of combined confidence
        conf_mask = (x[:, 4] >= conf_thres)
        x = x[conf_mask]
        
        # If none remain process next image
        if not x.shape[0]:
            continue
        
        # Convert boxes from [x, y, w, h] to [x1, y1, x2, y2]
        # Not needed if already in corner format, but leaving as is
        if True:
            box = xywh2xyxy(x[:, :4])
        else:
            box = x[:, :4]
            
        # Filter by class if needed
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            
        # Apply NMS
        if x.shape[0] > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        
        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = box, x[:, 4]  # boxes (offset by class), scores
        
        # Use original NMS function from utils.general if available
        try:
            from torchvision.ops import nms
            i = nms(boxes, scores, iou_thres)  # NMS
        except:
            # Fallback to manual NMS (much slower)
            i = torch.tensor([])
            if boxes.numel():
                # Sort by confidence
                sorted_indices = torch.argsort(scores, descending=True)
                boxes_sorted = boxes[sorted_indices]
                scores_sorted = scores[sorted_indices]
                
                # Iterate through boxes
                keep = []
                for j in range(len(boxes_sorted)):
                    if j in keep:
                        continue
                    keep.append(j)
                    # Get IoU with rest of boxes
                    remaining = list(range(j + 1, len(boxes_sorted)))
                    if remaining:
                        ious = box_iou(boxes_sorted[j:j+1], boxes_sorted[remaining])[0]
                        # Remove boxes with IoU > threshold
                        overlap = torch.where(ious > iou_thres)[0]
                        for k in overlap:
                            rem_idx = remaining[k]
                            if rem_idx not in keep:
                                keep.append(rem_idx)
                i = torch.tensor(keep)
        
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
            
        output[xi] = x[i]
        
    return output

# Load model
logger.info(f"Loading YOLOv5 model from {MODEL_PATH} on {DEVICE}...")
try:
    model = DetectMultiBackend(MODEL_PATH, device=DEVICE, dnn=False)
    stride = int(model.stride)
    names = model.names
    
    # Set confidence threshold much lower for class-only confidence
    conf_thres = float(os.getenv("YOLO_CONF", "0.5"))  # Higher threshold for class-only confidence
    iou_thres = float(os.getenv("YOLO_IOU", "0.45"))
    max_det = int(os.getenv("YOLO_MAX_DET", "100"))
    
    logger.info(f"✅ Loaded model: {len(names)} classes, stride={stride}")
    logger.debug(f"Classes: {names}")
    logger.info(f"Model config: conf_thres={conf_thres}, iou_thres={iou_thres}, max_det={max_det}")
    logger.info(f"Model class mapping: {[f'{i}: {name}' for i, name in enumerate(names)]}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    sys.exit(1)

# Set model to evaluation mode
model.eval()

# Function to get image URL from Cloudinary
def get_cloudinary_url(public_id):
    """Get secure URL from Cloudinary using public ID"""
    try:
        logger.debug(f"Getting Cloudinary URL for public_id: {public_id}")
        # Get resource details to ensure it exists
        resource = cloudinary.api.resource(public_id)
        if resource:
            # Generate secure URL
            url = cloudinary.utils.cloudinary_url(public_id, secure=True)[0]
            logger.debug(f"Generated Cloudinary URL: {url}")
            return url
        else:
            logger.error(f"Resource not found in Cloudinary: {public_id}")
            return None
    except Exception as e:
        logger.error(f"Error getting Cloudinary URL: {e}")
        return None

# Detection function
def detect_objects(image_url):
    try:
        logger.debug(f"Downloading image from URL: {image_url}")
        r = requests.get(image_url, stream=True, timeout=15)
        r.raise_for_status()
        img_arr = np.asarray(bytearray(r.content), dtype=np.uint8)
        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.error("Failed to decode image from bytes")
            return {"error": "Failed to decode image"}, 400

        # Preprocess: letterbox
        img, ratio, (dw, dh) = letterbox(frame, new_shape=640, stride=stride)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(DEVICE).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = model(img_tensor)

        # Handle list output
        if isinstance(pred, (list, tuple)):
            pred = pred[0]

        # NMS - custom class-only confidence
        det = custom_non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)[0]

        # Prepare predictions list
        predictions = []
        if det is not None and len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                cls_int = int(cls)
                label = names[cls_int] if isinstance(names, dict) else names[cls_int]
                box = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                predictions.append({
                    "label": label,
                    "confidence": float(conf),
                    "bbox": box
                })

        # Tampilkan beberapa prediksi dengan confidence tertinggi
        if predictions:
            # Urutkan prediksi berdasarkan confidence secara descending
            sorted_predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
            # Ambil 3 prediksi teratas (atau lebih sedikit jika tidak ada 3)
            top_predictions = sorted_predictions[:3]
            predictions = top_predictions
            count = len(top_predictions)
        else:
            count = 0

        return {
            "success": True,
            "predictions": predictions,
            "count": count,
            "model_info": {
                "conf_threshold": conf_thres,
                "iou_threshold": iou_thres
            }
        }

    except Exception as e:
        logger.error(f"Detection error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}, 500
    
# Fungsi untuk mendapatkan data permintaan baik dari JSON maupun form data
def get_request_data():
    """Get request data from either JSON or form data"""
    data = {}
    
    # Try to get JSON data
    if request.is_json:
        data = request.get_json(force=True)
    # If not JSON, try to get form data
    elif request.form:
        data = request.form.to_dict()
    # Try to get data from URL parameters
    elif request.args:
        data = request.args.to_dict()
    
    logger.debug(f"Request data: {data}")
    return data

# Routes
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "running",
        "model": "YOLOv5",
        "classes": list(names.values()) if isinstance(names, dict) else names,
        "device": str(DEVICE),
        "confidence_threshold": conf_thres
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    """Return detailed information about the loaded model"""
    try:
        return jsonify({
            "model_path": MODEL_PATH,
            "device": str(DEVICE),
            "stride": stride,
            "classes": names,
            "class_count": len(names),
            "conf_threshold": conf_thres,
            "iou_threshold": iou_thres,
            "max_detections": max_det
        })
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        start = time.time()
        # Get data from either JSON or form data
        data = get_request_data()
        
        # Support for direct URL input
        image_url = data.get('url')
        
        # Support for Cloudinary public_id input
        if not image_url:
            cloudinary_id = data.get('cloudinary_id')
            if cloudinary_id:
                image_url = get_cloudinary_url(cloudinary_id)
                if not image_url:
                    return jsonify({"error": "Invalid Cloudinary ID or resource not found"}), 400
        
        if not image_url:
            return jsonify({"error": "URL or Cloudinary ID parameter required"}), 400
        
        # Optional override for confidence threshold in request
        custom_conf = data.get('conf')
        if custom_conf is not None:
            global conf_thres
            old_conf = conf_thres
            try:
                conf_thres = float(custom_conf)
                logger.info(f"Using custom confidence threshold: {conf_thres} (was {old_conf})")
            except ValueError:
                logger.warning(f"Invalid confidence threshold value: {custom_conf}, using default: {old_conf}")
        
        logger.info(f"Processing request for: {image_url}")
        result = detect_objects(image_url)
        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        
        result["processing_time"] = f"{time.time() - start:.2f} seconds"
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in predict(): {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/cloudinary_resources', methods=['GET'])
def list_cloudinary_resources():
    """List available resources from Cloudinary"""
    try:
        # Get max_results parameter, default to 10
        max_results = request.args.get('max_results', '10')
        try:
            max_results = int(max_results)
        except ValueError:
            max_results = 10
        
        # Get resources from Cloudinary
        result = cloudinary.api.resources(max_results=max_results)
        
        # Format response
        resources = []
        for resource in result.get('resources', []):
            resources.append({
                "public_id": resource['public_id'],
                "format": resource['format'],
                "url": resource['url'],
                "secure_url": resource['secure_url'],
                "created_at": resource['created_at'],
                "bytes": resource['bytes'],
                "width": resource['width'],
                "height": resource['height'],
                "type": resource['resource_type']
            })
        
        return jsonify({
            "success": True,
            "count": len(resources),
            "resources": resources
        })
    except Exception as e:
        logger.error(f"Error listing Cloudinary resources: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"status": "ok", "message": "YOLOv5 API is running"})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    logger.info(f"Starting server on port {port}, debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)