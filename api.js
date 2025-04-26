const BASE_URL = "https://signatextbe-production.up.railway.app";

// -----------------------------
// Token Management
// -----------------------------
const getToken = () => localStorage.getItem("authToken");
const saveToken = (token) => localStorage.setItem("authToken", token);
const removeToken = () => localStorage.removeItem("authToken");

// -----------------------------
// AUTH
// -----------------------------
export const registerUser = async (username, email, password) => {
  try {
    const res = await fetch(`${BASE_URL}/api/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, email, password }),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Register failed: ${text}`);
    }

    const data = await res.json();
    if (data.token) saveToken(data.token);
    return data;
  } catch (err) {
    console.error("❌ Register error:", err);
    throw err;
  }
};

export const loginUser = async (username, email, password) => {
  try {
    const res = await fetch(`${BASE_URL}/api/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, email, password }),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Login failed: ${text}`);
    }

    const data = await res.json();
    if (data.token) saveToken(data.token);
    return data;
  } catch (err) {
    console.error("❌ Login error:", err);
    throw err;
  }
};

export const logoutUser = async () => {
  try {
    const res = await fetch(`${BASE_URL}/api/auth/logout`, {
      method: "POST",
      headers: { Authorization: `Bearer ${getToken()}` },
    });

    if (!res.ok) throw new Error("Logout failed");

    removeToken();
    return await res.json();
  } catch (err) {
    console.error("❌ Logout error:", err);
    throw err;
  }
};

export const getUserProfile = async () => {
  try {
    const res = await fetch(`${BASE_URL}/api/auth/me`, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${getToken()}`,
      },
    });

    if (!res.ok) throw new Error(`Failed to get user profile: ${res.statusText}`);

    return await res.json();
  } catch (error) {
    console.error("❌ Get user profile error:", error);
    throw error;
  }
};

export const getUserById = async (id) => {
  try {
    const res = await fetch(`${BASE_URL}/api/auth/user/${id}`, {
      method: "GET",
      headers: {
        Accept: "application/json",
        Authorization: `Bearer ${getToken()}`,
      },
    });

    if (!res.ok) throw new Error("Get user by ID failed");

    return await res.json();
  } catch (err) {
    console.error("❌ Get user by ID error:", err);
    throw err;
  }
};

// -----------------------------
// HISTORY
// -----------------------------
export const fetchAllHistory = async () => {
  try {
    const res = await fetch(`${BASE_URL}/api/history`, {
      method: "GET",
      headers: { Authorization: `Bearer ${getToken()}` },
    });

    if (!res.ok) throw new Error("Fetch all history failed");

    return await res.json();
  } catch (err) {
    console.error("❌ Fetch all history error:", err);
    throw err;
  }
};

export const fetchUserHistory = async (userId) => {
  try {
    const res = await fetch(`${BASE_URL}/api/history/user/${userId}`, {
      method: "GET",
      headers: { Authorization: `Bearer ${getToken()}` },
    });

    if (!res.ok) throw new Error("Fetch user history failed");

    return await res.json();
  } catch (err) {
    console.error("❌ Fetch user history error:", err);
    throw err;
  }
};

export const addHistory = async (user_id, translated_text) => {
  try {
    // Log payload and token for debugging
    console.log("Payload being sent:", { user_id, translated_text });
    console.log("Token being used:", getToken());

    const res = await fetch(`${BASE_URL}/api/history/save`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${getToken()}`,
      },
      body: JSON.stringify({ user_id, translated_text }),
    });

    // Check for non-OK responses
    if (!res.ok) {
      const errorDetails = await res.json();
      console.error("Server error details:", errorDetails);
      throw new Error(`Add history failed: ${errorDetails.message || "Unknown error"}`);
    }

    return await res.json();
  } catch (error) {
    console.error("❌ Add history error:", error);
    throw error;
  }
};

export const deleteHistory = async (id) => {
  try {
    const res = await fetch(`${BASE_URL}/api/history/${id}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${getToken()}` },
    });

    if (!res.ok) throw new Error("Delete history failed");

    return await res.json();
  } catch (err) {
    console.error("❌ Delete history error:", err);
    throw err;
  }
};

// -----------------------------
// UPLOAD MEDIA
// -----------------------------
export const uploadMediaFile = async (file) => {
  const formData = new FormData();
  formData.append("mediaFile", file);

  try {
    const res = await fetch(`${BASE_URL}/api/upload`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${getToken()}`,
      },
      body: formData,
    });

    if (!res.ok) throw new Error("Upload media failed");

    return await res.json();
  } catch (err) {
    console.error("❌ Upload media error:", err);
    throw err;
  }
};

// -----------------------------
// Export token utils
// -----------------------------
export { getToken, saveToken, removeToken };
