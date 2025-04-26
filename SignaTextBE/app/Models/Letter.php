<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Letter extends Model
{
    protected $fillable = [
        'user_id',
        'session_id',
        'letter',
        'accuracy',
    ];

    protected $casts = [
        'session_id' => 'string',
        'accuracy' => 'float',
    ];

    public function user()
    {
        return $this->belongsTo(User::class);
    }
}