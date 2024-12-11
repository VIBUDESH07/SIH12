import React from "react";
import logo1 from "../components/assets/logo1.png";
import logo2 from "../components/assets/logo2.png";
import { RiDashboardHorizontalFill } from "react-icons/ri";

const Header = () => {
  return (
    <div className="header">
      <div className="top-field">
        <img src={logo1} alt="logo" />
        <h2>Unique Identification Authority of India</h2>
        <img src={logo2} alt="logo" />
      </div>
      <div className="bottom-field">
        <div className="flex">
          <RiDashboardHorizontalFill className="icon" />
          <h3>myAadhaar</h3>
        </div>
      </div>
    </div>
  );
};

export default Header;
