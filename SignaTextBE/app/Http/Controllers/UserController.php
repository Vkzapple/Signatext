<?php

namespace App\Http\Controllers;

use App\Models\User;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;

class UserController extends Controller
{
    public function profile(Request $request)
    {
        // Ambil pengguna yang sedang login
        $loggedInUser = Auth::user();

        // Jika pengguna tidak terautentikasi, kembalikan respons 401 Unauthorized
        if (!$loggedInUser) {
            return response()->json([
                'success' => false,
                'message' => 'Unauthenticated',
            ], 401);
        }

        // Kembalikan profil pengguna yang sedang login
        return response()->json([
            'success' => true,
            'message' => 'User profile retrieved successfully',
            'data' => $loggedInUser,
        ], 200);
    }
}