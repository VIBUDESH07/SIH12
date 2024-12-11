import React from "react";
import { BrowserRouter, Route, Routes, useLocation } from "react-router-dom";
import Home from "./components/Home";
import Header from "./components/Header";
import Login from "./components/Login";
import LiveFace from "./components/LiveFace";
import LiveFace2 from "./components/LiveFace2";
import DashBoard from "./components/DashBoard";
import FaceLive from "./components/FaceLive";
import Final from "./components/Final";

function App() {
  const location = useLocation();
  return (
    <div>
      {location.pathname !== "/authenticate" && <Header />}
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/authenticate" element={<LiveFace2 />} />
        <Route path="/dashboard" element={<DashBoard />} />
      </Routes>
    </div>
  );
}

export default function Root() {
  return (
    <BrowserRouter>
      <App />
    </BrowserRouter>
  );
}
