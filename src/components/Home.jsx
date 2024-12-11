import React from "react";
import scanner from "../components/assets/scanner.gif";
import { Link } from "react-router-dom";

const Home = () => {
  return (
    <div className="home-container">
      <div className="banner">
        <div className="banner-content">
          <div className="b-left">
            <h1>
              Welcome to <span>myAadhaar</span>
            </h1>
            <p>
              Click on Login button to explore online demographics update
              service, Aadhaar PVC card ordering & tracking, and more
              value-added services offered by UIDAI. Your mobile number is
              required to be registered with the Aadhaar to login.
            </p>
          </div>
          <div className="b-right">
            <div className="scanner-field">
              <img src={scanner} alt="scanner" />
              <div className="login-field">
                <Link to="/login">
                  <button className="btn">Login</button>
                </Link>
                <p>Login with your face.</p>
              </div>
            </div>
          </div>
        </div>
        <div className="languages">
          <p>தமிழ்</p>
          <p>English</p>
          <p>हिंदी</p>
          <p>বাংলা</p>
          <p>ಕನ್ನಡ</p>
          <p>ગુજરાતી</p>
          <p>മലയാളം</p>
          <p>मराठी</p>
          <p>ଓଡ଼ିଆ</p>
          <p>ਪੰਜਾਬੀ</p>
          <p>తెలుగు</p>
          <p>اردو</p>
        </div>
      </div>
    </div>
  );
};

export default Home;
