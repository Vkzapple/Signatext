const BASE_URL = "https://signatextbe-production.up.railway.app"

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
    const res = await fetch(`${BASE_URL}/auth/register`, { 
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, email, password }),
    });

    if (!res.ok) throw new Error("Register failed");

    const data = await res.json();
    if (data.token) saveToken(data.token);
    return data;
  } catch (err) {
    console.error("❌ Register error:", err);
    throw err;
    const text = await res.text(); 
    try {
      const data = JSON.parse(text);
      if (data.token) saveToken(data.token);
      return data;
    } catch (err) {
      console.error("❌ Response bukan JSON:", text);
      throw new Error("Server mengembalikan halaman HTML, bukan JSON.");
    }
  } catch (error) {
    console.error("❌ Gagal register:", error);
    throw error;
  }
};

export const loginUser = async (username, email, password) => {
  try {
    const res = await fetch(`${BASE_URL}/api/auth/login`, {
    const res = await fetch(`${BASE_URL}/auth/login`, {  
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, email, password }),
    });

    if (!res.ok) throw new Error("Login failed");

    const data = await res.json();
    if (data.token) saveToken(data.token);
    return data;
  } catch (err) {
    console.error("❌ Login error:", err);
    throw err;
    const text = await res.text(); 
    try {
      const data = JSON.parse(text);
      if (data.token) saveToken(data.token);
      return data;
    } catch (err) {
      console.error("❌ Response bukan JSON:", text);
      throw new Error("Server mengembalikan halaman HTML, bukan JSON.");
    }
  } catch (error) {
    console.error("❌ Gagal login:", error);
    throw error;
  }
};

export const logoutUser = async () => {
  try {
    const res = await fetch(`${BASE_URL}/api/auth/logout`, {
    const res = await fetch(`${BASE_URL}/auth/logout`, {  
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

export async function getUserProfile(userId) {
  try {
    const encodedUserId = encodeURIComponent(userId);

    const res = await fetch(
      `https://signatextbe-production.up.railway.app/api/auth/user/${encodedUserId}`,
      {
        method: "GET",
        headers: {
          Authorization: `Bearer ${localStorage.getItem("authToken")}`,
        },
      }
    );

    if (!res.ok) {
      throw new Error(`Failed to get user profile: ${res.statusText}`);
    }

    const data = await res.json();
    return data;
  } catch (error) {
    console.error("❌ Get user profile error:", error);
    throw new Error("Get user profile failed");
  }
}

export const getUserById = async (id) => {
  try {
    const res = await fetch(`${BASE_URL}/api/auth/user/${id}`, {
    const res = await fetch(`${BASE_URL}/auth/me`, {  
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
  const res = await fetch(`${BASE_URL}/history`, {  
    method: "GET",
    headers: {
      Authorization: `Bearer ${getToken()}`,
    },
  });

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
  const res = await fetch(`${BASE_URL}/history/user/${userId}`, {  
    method: "GET",
    headers: {
      Authorization: `Bearer ${getToken()}`,
    },
  });

    return await res.json();
  } catch (err) {
    console.error("❌ Fetch user history error:", err);
    throw err;
  }
};

export async function addHistory(userId, resultText) {
  try {
    console.log("User ID:", userId);
    if (!userId) {
      throw new Error("User ID is undefined");
    }

    const res = await fetch(
      `https://signatextbe-production.up.railway.app/api/auth/user/${userId}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${localStorage.getItem("authToken")}`,
        },
        body: JSON.stringify({ resultText }),
      }
    );

    if (!res.ok) {
      throw new Error(`Failed to save history: ${res.statusText}`);
    }

    const data = await res.json();
    return data;
  } catch (error) {
    console.error("Error saving history:", error);
    alert("❌ Gagal menyimpan riwayat");
  }
}
export const addHistory = async (user_id, translated_text) => {
  const res = await fetch(`${BASE_URL}/history/save`, {  
    method: "POST",
    headers: getAuthHeaders(),
    body: JSON.stringify({ user_id, translated_text }),
  });

  return await res.json();
};

export const deleteHistory = async (id) => {
  try {
    const res = await fetch(`${BASE_URL}/api/history/${userId}`, {
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
  const res = await fetch(`${BASE_URL}/history/${id}`, {  
    method: "DELETE",
    headers: {
      Authorization: `Bearer ${getToken()}`,
    },
  });

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
