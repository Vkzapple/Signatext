const BASE_URL = "http://127.0.0.1:8000/api";

const getToken = () => localStorage.getItem("authToken");
const saveToken = (token) => localStorage.setItem("authToken", token);
const removeToken = () => localStorage.removeItem("authToken");

const getAuthHeaders = () => ({
  "Content-Type": "application/json",
  Authorization: `Bearer ${getToken()}`,
});

export const registerUser = async (username, email, password) => {
  try {
    const res = await fetch(`${BASE_URL}/auth/register`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ username, email, password }),
    });

    const text = await res.text(); // debug safe
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

export const loginUser = async (email, password) => {
  try {
    const res = await fetch(`${BASE_URL}/auth/login`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ email, password }),
    });

    const text = await res.text(); // debug
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
    const res = await fetch(`${BASE_URL}/auth/logout`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${getToken()}`,
      },
    });

    const data = await res.json();
    removeToken();
    return data;
  } catch (error) {
    console.error("❌ Gagal logout:", error);
    throw error;
  }
};

export const getUserProfile = async () => {
  try {
    const res = await fetch(`${BASE_URL}/auth/me`, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${getToken()}`,
      },
    });

    return await res.json();
  } catch (error) {
    console.error("❌ Gagal ambil profil:", error);
    throw error;
  }
};

export const fetchAllHistory = async () => {
  const res = await fetch(`${BASE_URL}/history`, {
    method: "GET",
    headers: {
      Authorization: `Bearer ${getToken()}`,
    },
  });

  return await res.json();
};

export const fetchUserHistory = async (userId) => {
  const res = await fetch(`${BASE_URL}/history/user/${userId}`, {
    method: "GET",
    headers: {
      Authorization: `Bearer ${getToken()}`,
    },
  });

  return await res.json();
};

export const addHistory = async (user_id, translated_text) => {
  const res = await fetch(`${BASE_URL}/history/save`, {
    method: "POST",
    headers: getAuthHeaders(),
    body: JSON.stringify({ user_id, translated_text }),
  });

  return await res.json();
};

export const deleteHistory = async (id) => {
  const res = await fetch(`${BASE_URL}/history/${id}`, {
    method: "DELETE",
    headers: {
      Authorization: `Bearer ${getToken()}`,
    },
  });

  return await res.json();
};

// Export token tools
export { getToken, saveToken, removeToken };
