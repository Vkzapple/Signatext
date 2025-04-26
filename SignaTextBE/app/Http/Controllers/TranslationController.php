<?php

namespace App\Http\Controllers;

use App\Models\Translation;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Validator;

class TranslationController extends Controller
{
    public function index()
    {
        $translations = Translation::with('user')->get();

        return response()->json([
            'success' => true,
            'data' => $translations,
        ], 200);
    }

    public function store(Request $request)
    {
        $validator = Validator::make($request->all(), [
            'user_id' => 'required',
            'translated_text' => 'required|string|max:65535',
        ]);

        if ($validator->fails()) {
            return response()->json([
                'success' => false,
                'message' => 'Validation failed',
                'errors' => $validator->errors(),
            ], 422);
        }

        $translation = Translation::create([
            'user_id' => $request->user_id,
            'translated_text' => $request->translated_text,
        ]);

        return response()->json([
            'success' => true,
            'message' => 'Translation created successfully',
            'data' => $translation,
        ], 201);
    }

    public function show($id)
    {
        $translation = Translation::with('user')->find($id);

        if (!$translation) {
            return response()->json([
                'success' => false,
                'message' => 'Translation not found',
            ], 404);
        }

        return response()->json([
            'success' => true,
            'data' => $translation,
        ], 200);
    }

    public function update(Request $request, $id)
    {
        // Validate the request data
        $validator = Validator::make($request->all(), [
            'translated_text' => 'nullable|string|max:65535',
        ]);

        if ($validator->fails()) {
            return response()->json([
                'success' => false,
                'message' => 'Validation failed',
                'errors' => $validator->errors(),
            ], 422);
        }

        // Find the translation
        $translation = Translation::find($id);

        if (!$translation) {
            return response()->json([
                'success' => false,
                'message' => 'Translation not found',
            ], 404);
        }

        // Update the translation
        $translation->update([
            'translated_text' => $request->translated_text ?? $translation->translated_text,
        ]);

        return response()->json([
            'success' => true,
            'message' => 'Translation updated successfully',
            'data' => $translation,
        ], 200);
    }

    public function destroy($id)
    {
        $translation = Translation::find($id);

        if (!$translation) {
            return response()->json([
                'success' => false,
                'message' => 'Translation not found',
            ], 404);
        }

        $translation->delete();

        return response()->json([
            'success' => true,
            'message' => 'Translation deleted successfully',
        ], 200);
    }

public function userHistory(Request $request, $userId)
    {
    // Get all translations for the specified user
    $translations = Translation::where('user_id', $userId)->get();

    return response()->json([
        'success' => true,
        'message' => 'User history retrieved successfully',
        'data' => $translations,
    ], 200);
    }
}