<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Str;

return new class extends Migration
{
    public function up()
    {
        Schema::create('letters', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->onDelete('cascade');
            $table->uuid('session_id')->default(\Illuminate\Support\Str::uuid());
            $table->char('letter', 1);
            $table->double('accuracy')->nullable(false);
            $table->timestamps();
        });

        DB::statement('ALTER TABLE letters ADD CONSTRAINT letters_accuracy_check CHECK (accuracy >= 0 AND accuracy <= 100)');
    }

    public function down()
    {
        Schema::dropIfExists('letters');
    }
};