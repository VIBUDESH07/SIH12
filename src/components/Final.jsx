import React, { useState, useRef, useEffect } from "react";
import Webcam from "react-webcam";
import { TiTick } from "react-icons/ti";
import { useNavigate, useLocation } from "react-router-dom";
import io from "socket.io-client";

const SERVER_URL = "https://172.21.4.242:5000";

export default function Final() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [result, setResult] = useState("Waiting for result...");
  const [borderColor, setBorderColor] = useState("gray");
  const [isCameraAllowed, setIsCameraAllowed] = useState(false);
  const [cameraLabel, setCameraLabel] = useState("");
  const [showOverlay, setShowOverlay] = useState(false);
  const [overlayMessage, setOverlayMessage] = useState("");
  const [currentInstruction, setCurrentInstruction] = useState("");
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [actionCounts, setActionCounts] = useState({});

  const navigate = useNavigate();
  const location = useLocation();
  const aadhaarNumber = location.state?.aadhaar;

  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "user",
  };

  const blockedCameras = [
    "Snap Camera",
    "OBS Virtual Camera",
    "ManyCam",
    "CamTwist",
    "XSplit VCam",
    "YouCam",
    "Ecamm Live",
    "vMix",
    "VirtualCam",
    "Webcamoid",
    "Fake Webcam",
    "AlterCam",
    "VidblasterX",
    "ChromaCam",
    "Logitech Capture",
  ];

  const updateBorderColor = (message) => {
    if (message.includes("FAKE")) {
      setBorderColor("red");
    } else if (message.includes("NOT MATCHING")) {
      setBorderColor("blue");
    } else if (message.includes("REAL") && message.includes("MATCHED")) {
      setBorderColor("#01b93b");
    } else {
      setBorderColor("gray");
    }
  };

  useEffect(() => {
    const socket = io(SERVER_URL);

    const sendFrame = () => {
      const canvas = canvasRef.current;
      const video = webcamRef.current?.video;

      if (!video || !canvas) return;

      const ctx = canvas.getContext("2d");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const frame = canvas.toDataURL("image/jpeg");

      if (frame) {
        socket.emit("send_frame", frame);
      } else {
        console.log("Failed to send frame.");
      }
    };

    socket.on("connect", () => {
      console.log("WebSocket connection established");
      if (aadhaarNumber) {
        socket.emit("send_aadhaar", { aadhaar: aadhaarNumber });
        console.log("Aadhaar number sent:", aadhaarNumber);
      }
    });

    socket.on("disconnect", () => {
      console.log("WebSocket connection closed");
    });

    socket.on("connect_error", (error) => {
      console.error("WebSocket connection error:", error);
    });

    socket.on("validate_face_matching", (data) => {
      const { status } = data;
      console.log(status);

      // Update border color and overlay message
      updateBorderColor(status);
      setOverlayMessage(status);

      //console.log(isAuthenticated, status);

      console.log(currentInstruction);

      // Navigate to dashboard if conditions are met
    });

    socket.on("receive_instruction", (data) => {
      const { instruction, action_counts } = data;
      console.log("Instruction:", instruction);
      console.log("Action Counts:", action_counts);

      // Update the current instruction
      setCurrentInstruction(instruction);

      // Store action_counts in state for external usage
      setActionCounts(action_counts);
    });

    socket.on("actions_completed", (data) => {
      const { status, message } = data;
      console.log("Actions Completed Status:", status, "Message:", message);

      if (status === "success") {
        setIsAuthenticated(true);
      }
    });

    const intervalId = setInterval(() => {
      if (isCameraAllowed) {
        sendFrame();
      }
    }, 1000);

    return () => {
      socket.off("connect");
      socket.off("disconnect");
      socket.off("connect_error");
      socket.off("validate_face_matching");
      socket.off("receive_instruction");
      socket.off("actions_completed");
      socket.disconnect();
      clearInterval(intervalId);
    };
  }, [aadhaarNumber, isCameraAllowed]);

  useEffect(() => {
    if (isAuthenticated && overlayMessage === "REAL and MATCHED") {
      setShowOverlay(true);
      console.log("Navigating to dashboard...");

      // Use a timer to delay navigation
      setTimeout(() => {
        navigate("/dashboard");
      }, 2000);
    }
  }, [isAuthenticated, overlayMessage, navigate]);

  const checkCameraAndAllow = async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(
        (device) => device.kind === "videoinput"
      );

      if (videoDevices.length > 0) {
        const cameraName = videoDevices[0].label;
        setCameraLabel(cameraName);

        const isVirtualCamera = blockedCameras.some((blockedName) =>
          cameraName.toLowerCase().includes(blockedName.toLowerCase())
        );

        setIsCameraAllowed(!isVirtualCamera);
        if (isVirtualCamera) {
          setResult("Virtual cameras are not allowed.");
          setBorderColor("red");
        }
      } else {
        setResult("No camera found.");
        setBorderColor("gray");
      }
    } catch (error) {
      console.error("Error checking camera:", error);
      setResult("Error detecting camera.");
      setBorderColor("gray");
    }
  };

  useEffect(() => {
    checkCameraAndAllow();
  }, []);

  return (
    <div className="face-authentication">
      <h1>Face Authentication</h1>
      {isCameraAllowed ? (
        <div
          style={{
            display: "inline-block",
            border: `10px solid ${borderColor}`,
            borderRadius: "20px",
          }}
        >
          <Webcam
            audio={false}
            height={480}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            width={640}
            videoConstraints={videoConstraints}
            style={{ transform: "scaleX(-1)" }}
          />
        </div>
      ) : (
        <p style={{ color: "red" }}>
          Virtual camera detected or no camera found. Please select a valid
          camera.
        </p>
      )}
      {/* <p>{result}</p> */}
      {cameraLabel && <p>Detected Camera: {cameraLabel}</p>}
      <h2>Instruction: {currentInstruction || "Waiting for server..."}</h2>
      {showOverlay && (
        <div className="overlay">
          <div className="success-container">
            <TiTick className="tick" />
            <p>Authentication Successfull</p>
          </div>
        </div>
      )}
      {isAuthenticated && <h2>Authentication Successful!</h2>}
      <canvas ref={canvasRef} style={{ display: "none" }} />
      <button onClick={checkCameraAndAllow}>Retry Camera Access</button>
    </div>
  );
}
