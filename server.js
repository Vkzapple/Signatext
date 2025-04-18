const express = require("express");
const { createProxyMiddleware } = require("http-proxy-middleware");
const path = require("path");

const app = express();
const PORT = 5000;

// Middleware untuk parsing body request
app.use(express.json());

// Serve static files
app.use("/assets", express.static(path.join(__dirname, "assets")));
app.use(express.static(path.join(__dirname)));

// Proxy untuk backend Laravel
app.use(
  "/auth",
  createProxyMiddleware({
    target: "http://localhost:8000",
    changeOrigin: true,
  })
);

// Route utama
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

app.listen(PORT, () => {
  console.log(`Server berjalan di http://localhost:${PORT}`);
});
