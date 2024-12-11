import React, { useState, useRef, useEffect } from "react";
import Webcam from "react-webcam";
import { io } from "socket.io-client";

const FaceLive = () => {
  const webcamRef = useRef(null);
  const [socket, setSocket] = useState(null);
  const [processedFrame, setProcessedFrame] = useState(null);
  const [message, setMessage] = useState("");
  const [instruction, setInstruction] = useState("");

  useEffect(() => {
    // Initialize the Socket.IO client
    const socketClient = io("http://localhost:5000");
    setSocket(socketClient);

    // Clean up on component unmount
    return () => {
      socketClient.disconnect();
    };
  }, []);

  useEffect(() => {
    if (socket) {
      // Listen for processed frames from the server
      socket.on("processed_frame", (data) => {
        setProcessedFrame(`data:image/jpeg;base64,${data.frame}`);
        setMessage(data.message);
        setInstruction(data.instruction);
      });
    }
  }, [socket]);

  const sendFrameToServer = async () => {
    if (webcamRef.current && socket) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        // Send the base64 frame to the server
        socket.emit("video_frame", { frame: imageSrc.split(",")[1] });
      }
    }
  };

  useEffect(() => {
    // Send frames at regular intervals
    const interval = setInterval(sendFrameToServer, 200); // 200ms interval
    return () => clearInterval(interval);
  }, [webcamRef, socket]);

  return (
    <div style={styles.container}>
      <h1>Face Liveness Detection</h1>
      <div style={styles.webcamContainer}>
        <div style={styles.webcam}>
          <h3>Live Feed</h3>
          <Webcam
            ref={webcamRef}
            audio={false}
            screenshotFormat="image/jpeg"
            style={styles.video}
          />
        </div>
        <div style={styles.webcam}>
          <h3>Processed Feed</h3>
          {processedFrame ? (
            <img src={processedFrame} alt="Processed frame" style={styles.video} />
          ) : (
            <p>Waiting for server...</p>
          )}
        </div>
      </div>
      <div style={styles.info}>
        <p>
          <strong>Instruction:</strong> {instruction || "Waiting..."}
        </p>
        <p>
          <strong>Message:</strong> {message || "Waiting..."}
        </p>
      </div>
    </div>
  );
};

const styles = {
  container: {
    textAlign: "center",
    fontFamily: "Arial, sans-serif",
    padding: "20px",
  },
  webcamContainer: {
    display: "flex",
    justifyContent: "space-around",
    marginTop: "20px",
  },
  webcam: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    border: "2px solid #ccc",
    borderRadius: "10px",
    padding: "10px",
    width: "45%",
  },
  video: {
    width: "100%",
    height: "auto",
    borderRadius: "10px",
  },
  info: {
    marginTop: "20px",
    textAlign: "left",
    padding: "0 10%",
  },
};

export default FaceLive;
