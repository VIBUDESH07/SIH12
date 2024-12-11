import React, { useState, useRef, useEffect } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { TiTick } from "react-icons/ti";

const LiveFace = () => {
  const webcamRef = useRef(null);
  const [result, setResult] = useState("Waiting for result...");
  const [borderColor, setBorderColor] = useState("gray");
  const [isCameraAllowed, setIsCameraAllowed] = useState(false);
  const [cameraLabel, setCameraLabel] = useState("");
  const [showOverlay, setShowOverlay] = useState(false);
  const [overlayMessage, setOverlayMessage] = useState("");
  const [timer, setTimer] = useState(0); // Timer for success duration
  const [intervalId, setIntervalId] = useState(null);
  const navigate = useNavigate();

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
    } else if (message.includes("REAL but NOT MATCHED")) {
      setBorderColor("blue");
    } else if (message.includes("REAL") && message.includes("MATCHED")) {
      setBorderColor("#01b93b");
    } else {
      setBorderColor("gray");
    }
  };

  const sendImageToBackend = (imageSrc) => {
    axios
      .post("http://localhost:5000/api/process-image", {
        image: imageSrc,
      })
      .then((response) => {
        const message = response.data.message;
        setResult(message);
        updateBorderColor(message);
  
        // If message is "REAL and MATCHED", start the timer
        if (message.includes("REAL") && message.includes("MATCHED")) {
          // Start a timer for 5 seconds
          if (intervalId) clearInterval(intervalId); // Clear any existing interval
          setTimer(0); // Reset the timer
          const newIntervalId = setInterval(() => {
            setTimer((prevTimer) => {
              if (prevTimer >= 50) {
            
                clearInterval(newIntervalId);
                setShowOverlay(true);
                setOverlayMessage("Authentication Successful");
                setTimeout(() => navigate("/dashboard"), 2000); // Redirect after 2 seconds
              }
              return prevTimer + 1;
            });
          }, 1000); // Increase timer by 1 second each interval
          setIntervalId(newIntervalId);
        } else {
          // If the result is not "REAL and MATCHED", reset the timer
          if (intervalId) clearInterval(intervalId); // Clear the interval if it's running
          setTimer(0); // Reset the timer
          setShowOverlay(false); // Hide the overlay
        }
      })
      .catch((error) => {
        console.error("Error sending image to backend:", error);
        setResult("Error processing image.");
        setBorderColor("gray");
        setShowOverlay(false);
      });
  };
  

  const captureFrame = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc) {
      sendImageToBackend(imageSrc);
    }
  };

  useEffect(() => {
    const id = setInterval(() => {
      if (isCameraAllowed) {
        captureFrame();
      }
    }, 600); // Capture every 600 ms (adjust as needed)
    setIntervalId(id);

    return () => clearInterval(id); // Cleanup interval
  }, [isCameraAllowed]);

  const checkCameraAndAllow = async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(
        (device) => device.kind === "videoinput"
      );

      if (videoDevices.length > 0) {
        const cameraName = videoDevices[0].label; // Use the first available camera

        setCameraLabel(cameraName);

        const isVirtualCamera = blockedCameras.some((blockedName) =>
          cameraName.toLowerCase().includes(blockedName.toLowerCase())
        );

        if (!isVirtualCamera) {
          setIsCameraAllowed(true);
        } else {
          setIsCameraAllowed(false);
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
      <p style={{ fontSize: "1.2rem", fontWeight: "600" }}>{result}</p>
      {cameraLabel && <p>Detected Camera: {cameraLabel}</p>}

      {/* Overlay for authentication success */}
      {showOverlay && (
        <div className="overlay">
          <div className="success-container">
            <TiTick className="tick" />
            <p>{overlayMessage}</p>
          </div>
        </div>
      )}

      {/* Display timer for debugging */}
      <p>Timer: {timer} seconds</p>
    </div>
  );
};

export default LiveFace;
