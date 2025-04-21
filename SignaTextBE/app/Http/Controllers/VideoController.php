<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Cloudinary\Cloudinary;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Validator;

class VideoController extends Controller
{
    public function uploadMedia(Request $request)
    {
        try {
            // Validasi input
            $validator = Validator::make($request->all(), [
                'mediaFile' => 'required|file|mimes:mp4,mov,ogg,qt,jpg,jpeg,png,gif|max:102400', // Maksimal 100MB
            ], [
                'mediaFile.required' => 'A media file is required.',
                'mediaFile.mimes' => 'The file must be a video (mp4, mov, ogg, qt) or image (jpg, jpeg, png, gif).',
                'mediaFile.max' => 'The file size must not exceed 100MB.',
            ]);

            if ($validator->fails()) {
                return response()->json(['errors' => $validator->errors()], 422);
            }

            // Logging detail file yang diupload
            $uploadedFile = $request->file('mediaFile');
            Log::info('Uploaded file details:', [
                'name' => $uploadedFile->getClientOriginalName(),
                'mime' => $uploadedFile->getMimeType(),
                'size' => $uploadedFile->getSize(),
            ]);

            // Konfigurasi Cloudinary
            $cloudinary = new Cloudinary([
                'cloud' => [
                    'cloud_name' => env('CLOUDINARY_CLOUD_NAME'),
                    'api_key' => env('CLOUDINARY_API_KEY'),
                    'api_secret' => env('CLOUDINARY_API_SECRET'),
                ],
            ]);

            // Upload file ke Cloudinary
            $uploadResult = $cloudinary->uploadApi()->upload($uploadedFile->getRealPath(), [
                'folder' => 'media-uploads', // Folder untuk mengelompokkan file
                'resource_type' => 'auto', // Otomatis deteksi tipe file (image/video)
            ]);

            // Dapatkan URL file yang diunggah
            $mediaUrl = $uploadResult['secure_url'];
            Log::info('File uploaded to Cloudinary:', ['url' => $mediaUrl]);

            // Kirim URL file ke API Computer Vision untuk diproses
            $result = $this->processMediaWithComputerVision($mediaUrl);

            return response()->json(['result' => $result], 200);
        } catch (\Exception $e) {
            Log::error('Error uploading media: ' . $e->getMessage());
            return response()->json(['error' => 'An error occurred while uploading the media.'], 500);
        }
    }

    private function processMediaWithComputerVision($mediaUrl)
    {
        try {
            $endpoint = 'https://enthusiastic-adventure-production.up.railway.app/predict';

            // Logging request ke API Computer Vision
            Log::info('Sending request to Computer Vision API:', ['url' => $mediaUrl]);

            $response = Http::withHeaders([
                'Content-Type' => 'application/x-www-form-urlencoded',
            ])->post($endpoint, [
                'url' => $mediaUrl,
            ]);
            
            // Log payload yang dikirim
            Log::info('Payload sent to Flask:', ['payload' => ['url' => $mediaUrl]]);
            
            // Log respons dari Flask
            Log::info('API response:', ['status' => $response->status(), 'body' => $response->body()]);

            if ($response->failed()) {
                Log::error('API processing error: ' . $response->status() . ' - ' . $response->body());
                throw new \Exception('Failed to process media with Computer Vision API. Status: ' . $response->status());
            }

            return $response->json();
        } catch (\Exception $e) {
            Log::error('Error processing media with Computer Vision API: ' . $e->getMessage());
            throw $e;
        }
    }
}