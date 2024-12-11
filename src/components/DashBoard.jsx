import React from "react";
import card from "../components/assets/aadhaar.webp";
import Footer from "./Footer";

const DashBoard = () => {
  return (
    <div className="dashboard">
      <div className="welcome-field">
        <h1>
          Welcome to myAadhaar <span>"Vanguards Elevate"</span>
        </h1>
      </div>
      <div className="aadhaar-card">
        <img src={card} alt="card" className="card" />
      </div>
      <Footer />
    </div>
  );
};

export default DashBoard;
