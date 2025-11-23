import { auth } from "./firebase-config.js";
import { 
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  sendEmailVerification,
  sendPasswordResetEmail,
  signOut,
  onAuthStateChanged
} from "https://www.gstatic.com/firebasejs/9.23.0/firebase-auth.js";

// Global flag to prevent loops
let isCheckingAuth = false;
let isInitialAuthCheck = true;

// Only run auth check if Firebase auth is available
if (auth) {
  auth.onAuthStateChanged((user) => {
    if (isCheckingAuth) return;
    isCheckingAuth = true;

    try {
      const currentPath = window.location.pathname;
      const isAuthPage = currentPath.includes("/auth");

      if (user) {
        if (!user.emailVerified) {
          signOut(auth);
          if (!isAuthPage) {
            window.location.href = "/auth/login";
          }
          return;
        }

        if (
          isAuthPage &&
          !currentPath.includes("/auth/mfa") &&
          !isInitialAuthCheck
        ) {
          window.location.href = "/dashboard";
        }
      } else {
        if (!isAuthPage && !isInitialAuthCheck) {
          window.location.href = "/auth/login";
        }
      }
    } finally {
      isCheckingAuth = false;
      isInitialAuthCheck = false;
    }
  });
}

// ==== Auth Functions ====

const handleRegister = async (email, password) => {
  try {
    const userCredential = await createUserWithEmailAndPassword(
      auth,
      email,
      password
    );
    await sendEmailVerification(userCredential.user);
    return userCredential;
  } catch (error) {
    throw error;
  }
};

const handleLogin = async (email, password) => {
  try {
    const userCredential = await signInWithEmailAndPassword(
      auth,
      email,
      password
    );
    if (!userCredential.user.emailVerified) {
      await signOut(auth);
      throw new Error("Please verify your email first");
    }
    return userCredential;
  } catch (error) {
    throw error;
  }
};

const handlePasswordReset = async (email) => {
  try {
    await sendPasswordResetEmail(auth, email);
    return true;
  } catch (error) {
    throw error;
  }
};

// ==== DOM Ready ====
document.addEventListener("DOMContentLoaded", () => {
  // Login Form
  const loginForm = document.getElementById("loginForm");
  if (loginForm) {
    loginForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const email = e.target.email.value;
      const password = e.target.password.value;

      try {
        await handleLogin(email, password);
        window.location.href = "/auth/mfa";
      } catch (error) {
        alert(error.message);
      }
    });
  }

  // Registration Form
  const registerForm = document.getElementById("registerForm");
  if (registerForm) {
    registerForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const email = e.target.email.value;
      const password = e.target.password.value;

      try {
        await handleRegister(email, password);
        alert(
          "Registration successful! Please check your email for verification."
        );
        window.location.href = "/auth/login";
      } catch (error) {
        alert(error.message);
      }
    });
  }

  // MFA Form
  const mfaForm = document.getElementById("mfaForm");
  if (mfaForm) {
    mfaForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const otp = e.target.otp.value;

      if (otp === "123456") {
        try {
          const user = auth.currentUser;
          if (!user) throw new Error("No authenticated user");

          const token = await user.getIdToken();
          localStorage.setItem("firebaseToken", token);

          const cookieResponse = await fetch("/set_auth_cookie", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${token}`,
            },
            credentials: "include",
          });

          if (!cookieResponse.ok) {
            const errorData = await cookieResponse.json();
            throw new Error(errorData.message || "Failed to set auth cookie");
          }

          localStorage.removeItem("firebaseToken");

          window.location.href = "/dashboard";
        } catch (error) {
          localStorage.removeItem("firebaseToken");
          console.error("MFA verification or cookie setting failed:", error);
          alert("Authentication error: " + error.message);
        }
      } else {
        alert("Invalid OTP code");
      }
    });
  }

  // Forgot Password Form
  const forgotForm = document.getElementById("forgotForm");
  if (forgotForm) {
    forgotForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const email = e.target.email.value;

      try {
        await handlePasswordReset(email);
        alert("Password reset link sent to your email");
      } catch (error) {
        alert(error.message);
      }
    });
  }

  // Trigger first token refresh on load
  refreshTokenHourly();
});

// ==== Token Refresh ====
const refreshTokenHourly = () => {
  if (auth && auth.currentUser) {
    auth.currentUser
      .getIdToken(true)
      .then((token) => {
        localStorage.setItem("firebaseToken", token);
        console.log("Token refreshed and saved to localStorage.");
      })
      .catch((error) => {
        console.error("Token refresh error:", error);
      });
  } else {
    console.warn("No user is currently signed in. Skipping token refresh.");
  }
};

// Set interval for future token refreshes
setInterval(refreshTokenHourly, 60 * 60 * 1000); // every 1 hour
