<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Translation extends Model
{
    protected $fillable = [
        'user_id',
        'translated_text',
    ];

    public function user()
    {
        return $this->belongsTo(User::class);
    }
}