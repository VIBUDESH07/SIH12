import React, { useState } from "react";
import Footer from "./Footer";
import { useNavigate } from "react-router-dom";

const Login = () => {
  const [aadhaar, setAadhaar] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const navigate = useNavigate();

  const handleInputChange = (e) => {
    const value = e.target.value;
    // Allow only numbers
    if (/^\d*$/.test(value)) {
      setAadhaar(value);
      // Clear error if input is valid
      setErrorMessage("");
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Check if the number is 12 digits
    if (aadhaar.length !== 12) {
      setErrorMessage("Aadhaar number must be exactly 12 digits.");
    } else {
      setErrorMessage("");
      navigate("/authenticate", { state: { aadhaar } });
    }
  };

  return (
    <div className="login-container">
      <h3>Login to Aadhaar via FACE</h3>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Enter your Aadhaar card number"
          value={aadhaar}
          onChange={handleInputChange}
          maxLength="12"
        />
        <div className="error-message">{errorMessage}</div>
        <button type="submit" className="btn">
          Login with face
        </button>
      </form>
      <Footer />
    </div>
  );
};

export default Login;
