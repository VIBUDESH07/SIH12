import React, { useState, useRef, useEffect } from "react";
import Webcam from "react-webcam";
import { useNavigate, useLocation } from "react-router-dom";
import { TiTick } from "react-icons/ti";

const LiveFaceWebSocket = () => {
  const webcamRef = useRef(null);
  const [result, setResult] = useState("Waiting for result...");
  const [borderColor, setBorderColor] = useState("gray");
  const [isCameraAllowed, setIsCameraAllowed] = useState(false);
  const [cameraLabel, setCameraLabel] = useState("");
  const [showOverlay, setShowOverlay] = useState(false);
  const [overlayMessage, setOverlayMessage] = useState("");
  const [ws, setWs] = useState(null);
  const navigate = useNavigate();
  const location = useLocation(); // Access location state
  const [matchedCounter, setMatchedCounter] = useState(0);

  const aadhaarNumber = location.state?.aadhaar; 

  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "user",
  };

  const blockedCameras = ["Snap Camera", "OBS Virtual Camera"];

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

  const captureFrame = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc && ws) {
      ws.emit("process_image", { image: imageSrc });
    }
  };

  useEffect(() => {
    const intervalId = setInterval(() => {
      if (isCameraAllowed && ws) {
        captureFrame();
      }
    }, 1000);
    return () => clearInterval(intervalId);
  }, [isCameraAllowed, ws]);

  useEffect(() => {
    const socket = require("socket.io-client")("http://172.21.4.242:5000");
    setWs(socket);

    socket.on("connect", () => {
        console.log("WebSocket connection established");

        if (aadhaarNumber) {
            socket.emit("aadhaar_number", { aadhaar: aadhaarNumber });
            console.log("Aadhaar number sent:", aadhaarNumber);
        }
    });

    socket.on("response", (data) => {
        const message = data.message;
        setResult(message);
        updateBorderColor(message);

        if (message.includes("REAL") && message.includes("MATCHED")) {
            setMatchedCounter((prevCounter) => {
                const newCounter = prevCounter + 1;
                if (newCounter >= 4) {
                    setShowOverlay(true);
                    setOverlayMessage("Authentication Successful");
                }
                return newCounter;
            });
        } else {
            setMatchedCounter(0);
            setShowOverlay(false);
            setOverlayMessage("");
        }
    });

    socket.on("disconnect", () => {
        console.log("WebSocket connection closed");
    });

    socket.on("connect_error", (error) => {
        console.error("WebSocket connection error:", error);
    });

    return () => {
        socket.close();
    };
}, [aadhaarNumber]);


  useEffect(() => {
    if (showOverlay && overlayMessage === "Authentication Successful") {
      const timer = setTimeout(() => {
        navigate("/dashboard");
      }, 2000);

      return () => clearTimeout(timer);
    }
  }, [showOverlay, overlayMessage, navigate]);

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
      <h2>Face Detection</h2>
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
          />
        </div>
      ) : (
        <p style={{ color: "red" }}>
          Virtual camera detected or no camera found. Please select a valid
          camera.
        </p>
      )}
      <p>{result}</p>
      {cameraLabel && <p>Detected Camera: {cameraLabel}</p>}

      {showOverlay && (
        <div className="overlay">
          <div className="success-container">
            <TiTick className="tick" />
            <p>{overlayMessage}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default LiveFaceWebSocket;
