# SignaText - Web Application for Sign Language Translation

SignaText is a web application that translates sign language into text by uploading a video. Initially designed for real-time translation, the project now allows users to upload sign language videos for processing. The app leverages machine learning models and AI to recognize hand movements (sign language) and translate them into text.

## Features
- Upload a sign language video for processing.
- Translate sign language into text using AI (YOLO).
- Store the translation results in a PostgreSQL database.
- User authentication for secure access to features.

## Technologies Used
- **Frontend**: HTML, CSS, JavaScript (No framework like React/Svelte)
- **Backend**: PHP, Laravel, deployed on Railway
- **Machine Learning**: YOLO, deployed on Railway
- **Database**: PostgreSQL
- **Cloud Storage**: Cloudinary (for video hosting)

## Getting Started
Follow these steps to get a copy of the project running on your local machine for development and testing purposes.

### Prerequisites
Make sure you have the following installed:
- **PHP** (for Laravel): [Install PHP](https://www.php.net/downloads.php)
- **Composer** (PHP package manager): [Install Composer](https://getcomposer.org/)
- **Node.js**: [Install Node.js](https://nodejs.org/)
- **Python 3.x**: [Install Python](https://www.python.org/)
- **PostgreSQL**: [Install PostgreSQL](https://www.postgresql.org/download/)

### Clone the Repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/yourusername/signatext.git
cd signatext
```

### Setting Up the Backend (Laravel)
1. Install the required PHP dependencies:
   ```bash
   cd backend
   composer install
   ```

2. Set up environment variables:
   Copy `.env.example` to `.env` and update the necessary values (e.g., database credentials, API keys):
   ```bash
   cp .env.example .env
   ```

3. Generate the application key:
   ```bash
   php artisan key:generate
   ```

4. Run database migrations:
   ```bash
   php artisan migrate
   ```

5. Start the Laravel server:
   ```bash
   php artisan serve
   ```

### Setting Up the Frontend
1. In the frontend folder, make sure all necessary files are in place (`index.html`, `style.css`, `script.js`), and the structure is correctly set for your needs.

2. If you're using plain HTML, CSS, and JavaScript, make sure your files are correctly linked and the routing is configured to work with your backend.

3. If deploying on a platform like **Vercel**, ensure to connect your frontend repository to the platform for deployment.

### Deploying the Machine Learning Model (YOLO)
1. Deploy the machine learning model (YOLO) on Railway:
   - Create an account on [Railway](https://railway.app/).
   - Deploy your YOLO-based AI model on Railway following their documentation.
   - Make sure to update the endpoint URL of the model in the backend configuration.

### Running the Application

Once both the frontend and backend are running, you can access the app in your browser:

- Frontend (deployed URL, e.g., Vercel)
- Backend: `http://localhost:8000`

### Uploading Video and Translating Sign Language
1. Log in to the application.
2. Go to the "Upload" page.
3. Upload a video of sign language.
4. The app will process the video, using YOLO to translate the sign language into text.
5. The translation results will be stored in the PostgreSQL database.

## Project Structure
```
/backend
  /app
    /Controllers
    /Models
  /config
  /database
    /migrations
/frontend
  /index.html
  /style.css
  /script.js
```

## Troubleshooting
- **Cannot connect to the database**: Ensure that your PostgreSQL configuration in `.env` is correct.
- **Video upload fails**: Make sure that your Cloudinary account is set up correctly and that the API key is added to the `.env` file.
- **AI model not working**: Verify that your YOLO model endpoint is correct and deployed on Railway.

## Contributing
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to your branch (`git push origin feature-branch`).
6. Create a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
If you have any questions, feel free to contact us at [evellykhnz@gmail.com].
