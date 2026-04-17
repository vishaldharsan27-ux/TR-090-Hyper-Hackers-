import { useEffect, useRef, useState } from "react";
import * as tf from '@tensorflow/tfjs';
import { HandLandmarker, PoseLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";
import { db } from './firebase'; // Your existing firebase config
import { collection, addDoc, serverTimestamp } from 'firebase/firestore';
import labels from './assets/labels.json'; // Ensure file is in src/assets/
import { jsPDF } from 'jspdf'; // newly installed PDF generator

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [prediction, setPrediction] = useState("Waiting...");
  const [confidence, setConfidence] = useState(0);
  const [sector, setSector] = useState("General");
  const [transcript, setTranscript] = useState<string[]>([]); // Collect session words
  
  // NEW STATES: Bounded Sequence recording and Handedness Swapping
  const [isRecording, setIsRecording] = useState(false);
  const [swapHands, setSwapHands] = useState(false);
  const recordingActive = useRef(false);
  const swapRef = useRef(false);

  // 1. Core AI References
  const handLandmarker = useRef<HandLandmarker | null>(null);
  const poseLandmarker = useRef<PoseLandmarker | null>(null);
  const signModel = useRef<tf.LayersModel | null>(null);
  const frameBuffer = useRef<number[][]>([]); // 30-frame sequence buffer

  // Sync state refs to bypass closure staleness
  useEffect(() => { swapRef.current = swapHands; }, [swapHands]);
  useEffect(() => { recordingActive.current = isRecording; }, [isRecording]);

  useEffect(() => {
    const setupAI = async () => {
      // Load MediaPipe Landmarkers
      const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
      
      handLandmarker.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: "/hand_landmarker.task" }, // Must be in public/
        runningMode: "VIDEO",
        numHands: 2
      });

      poseLandmarker.current = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: "/pose_landmarker.task" }, // Must be in public/
        runningMode: "VIDEO",
        numPoses: 1
      });

      // Load your trained LSTM Brain
      try {
        signModel.current = await tf.loadLayersModel('/tfjs_model/model.json'); // Must be in public/
      } catch (err) {
        console.warn("Could not load model.json. Ensure it is placed in public/tfjs_model/", err);
      }

      startCamera();
    };

    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
            videoRef.current.onloadeddata = () => {
              videoRef.current?.play();
              predictLoop();
            };
        }
      } catch (err) {
        console.error("Error accessing camera", err);
      }
    };

    setupAI();
  }, []);

  // Normalize mapping (wrist to 0) helper
  const normalizeHand = (arr: number[]) => {
      if (!arr.some(v => v !== 0)) return arr;
      const wristX = arr[0], wristY = arr[1], wristZ = arr[2];
      return arr.map((val, i) => {
          if (i % 3 === 0) return val - wristX;
          if (i % 3 === 1) return val - wristY;
          return val - wristZ;
      });
  };

  // 2. The Real-Time Inference Loop
  const lastCaptureTime = useRef<number>(0);

  const predictLoop = async () => {
    // Only require video and marker models to draw points
    if (videoRef.current && canvasRef.current && handLandmarker.current && poseLandmarker.current && videoRef.current.readyState >= 2) {
      const startTimeMs = performance.now();
      const handResults = handLandmarker.current.detectForVideo(videoRef.current, startTimeMs);
      const poseResults = poseLandmarker.current.detectForVideo(videoRef.current, startTimeMs);

      // --- 1. Drawing Context Configuration ---
      const ctx = canvasRef.current.getContext("2d");
      if (ctx) {
          ctx.save();
          ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
          const drawingUtils = new DrawingUtils(ctx);
          
          if (poseResults.landmarks) {
             for (const landmarks of poseResults.landmarks) {
                 drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: "white", lineWidth: 2 });
                 drawingUtils.drawLandmarks(landmarks, { color: "red", lineWidth: 1, radius: 2 });
             }
          }
          if (handResults.landmarks) {
             for (const landmarks of handResults.landmarks) {
                 drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 4 });
                 drawingUtils.drawLandmarks(landmarks, { color: "#FF0000", lineWidth: 2, radius: 3 });
             }
          }
          ctx.restore();
      }

      // --- 2. Temporal Sampling (Matches Python training: interval=5 -> ~6 FPS) ---
      // We ONLY extract parameters and push to the buffer when recording is ACTIVE to mimic bounded training videos
      if (recordingActive.current && startTimeMs - lastCaptureTime.current >= 166) {
          lastCaptureTime.current = startTimeMs;

          // Feature Extraction Pipeline (matching python 30x225 format)
          let poseVec = new Array(99).fill(0); // 33 * 3
          let leftHandVec = new Array(63).fill(0); // 21 * 3
          let rightHandVec = new Array(63).fill(0); // 21 * 3

          if (poseResults.landmarks && poseResults.landmarks.length > 0) {
             poseVec = poseResults.landmarks[0].flatMap((l: any) => [l.x, l.y, l.z]);
          }

          if (handResults.landmarks && handResults.handedness) {
              handResults.handedness.forEach((list, i) => {
                 let category = list[0].categoryName;
                 
                 // Fix missing/reversed Dataset handedness if their models trained on mirrored video!
                 if (swapRef.current) {
                     category = (category === "Left") ? "Right" : "Left";
                 }

                 const arr = handResults.landmarks[i].flatMap((l: any) => [l.x, l.y, l.z]);
                 if (category === "Left") leftHandVec = arr;
                 else rightHandVec = arr;
              });
          }

          leftHandVec = normalizeHand(leftHandVec);
          rightHandVec = normalizeHand(rightHandVec);

          const combinedVec = [...poseVec, ...leftHandVec, ...rightHandVec]; // Length 225
          frameBuffer.current.push(combinedVec);

          // Update recording UI percentage (Debug visualization)
          const bufferPct = Math.round((frameBuffer.current.length / 30) * 100);
          setPrediction(`Recording [${bufferPct}%]...`);

          // Once we reach 30 frames naturally, evaluate and stop recording
          if (frameBuffer.current.length === 30) {
              setIsRecording(false);
              evaluateBuffer();
          }
      }
    }
    requestAnimationFrame(predictLoop);
  };

  const evaluateBuffer = async () => {
      if (!signModel.current || frameBuffer.current.length !== 30) return;
      
      try {
        const tensorInput = tf.tensor3d([frameBuffer.current]); // Shape [1, 30, 225]
        const predictionResult = signModel.current.predict(tensorInput) as tf.Tensor;
        const scores = await predictionResult.data();
        const maxScore = Math.max(...Array.from(scores));
        const index = scores.indexOf(maxScore);

        const word = Object.keys(labels).find(key => (labels as any)[key] === index) || "Unknown";
        
        setPrediction(word);
        setConfidence(Math.round(maxScore * 100));

        // If it passes 50% threshold, it is considered a valid sign capture
        if (maxScore > 0.50) { 
            triggerOutputs(word);
        }
        
        tensorInput.dispose();
        predictionResult.dispose();
      } catch(e) {
         console.error("Prediction error", e);
         signModel.current = null;
      }
      frameBuffer.current = []; // Clear for next explicit record session
  };

  const handleRecordSubmit = () => {
      frameBuffer.current = []; // Fresh sequence
      setIsRecording(true);
  };

  const handleStopRecording = () => {
      setIsRecording(false);
      // Zero pad the remainder to perfectly match short training datasets!
      const paddingNeeded = 30 - frameBuffer.current.length;
      for(let i = 0; i < paddingNeeded; i++) {
          frameBuffer.current.push(new Array(225).fill(0));
      }
      evaluateBuffer();
  };

  // 3. Multimodal Output Logic
  const triggerOutputs = (word: string) => {
    // Build Local Transcript
    setTranscript(prev => [...prev, word]);

    // Text-to-Speech (Web Speech API Fallback)
    const utterance = new SpeechSynthesisUtterance(word);
    utterance.lang = 'hi-IN'; // Indian Accent
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);

    // Save to Firebase Database
    try {
        addDoc(collection(db, "sessions"), {
            word,
            sector,
            timestamp: serverTimestamp()
        });
    } catch(err) {
        console.warn("Firebase not configured", err);
    }
  };

  // 4. Download PDF Functionality
  const downloadPDF = () => {
    const doc = new jsPDF();
    doc.setFontSize(22);
    doc.text("Session Transcript", 20, 20);
    
    doc.setFontSize(12);
    doc.setTextColor(100);
    doc.text(`Sector/Mode: ${sector}`, 20, 30);
    doc.text(`Date & Time: ${new Date().toLocaleString()}`, 20, 38);
    
    doc.setLineWidth(0.5);
    doc.line(20, 42, 190, 42);

    doc.setFontSize(16);
    doc.setTextColor(0);
    let y = 52;

    if (transcript.length === 0) {
        doc.text("No signs detected in this session yet.", 20, y);
    } else {
        transcript.forEach((word, index) => {
            doc.text(`${index + 1}. ${word}`, 20, y);
            y += 10;
            // Native Pagination Support
            if (y > 280) {
                doc.addPage();
                y = 20;
            }
        });
    }

    doc.save("Sign_Transcript.pdf");
  };

  return (
    <div className="flex flex-col items-center p-8 bg-gray-900 min-h-screen text-white">
      <h1 className="text-4xl font-bold mb-4 text-blue-400">Communication Bridge</h1>

      <div className="flex gap-4 mb-4">
          <select
            onChange={(e) => setSector(e.target.value)}
            className="p-2 bg-gray-800 rounded border border-blue-500"
          >
            <option value="General">General Mode</option>
            <option value="Hospital">Hospital Mode</option>
            <option value="Bank">Bank Mode</option>
            <option value="Govt">Government Office</option>
          </select>

          <label className="flex items-center gap-2 bg-gray-800 p-2 rounded border border-yellow-600 text-yellow-500 cursor-pointer">
              <input type="checkbox" checked={swapHands} onChange={(e) => setSwapHands(e.target.checked)} />
              Fix Mirrored Model (Swap Handedness)
          </label>
      </div>

      <div className="relative border-4 border-blue-500 rounded-lg overflow-hidden w-full max-w-[800px] aspect-auto">
        <video 
          ref={videoRef} 
          autoPlay 
          playsInline 
          className="w-full object-contain bg-black scale-x-[-1]" 
        />
        <canvas 
          ref={canvasRef} 
          width="1280" 
          height="720" 
          className="absolute inset-0 w-full h-full object-contain pointer-events-none scale-x-[-1]" 
        />
        
        <div className="absolute top-4 left-4 bg-black/50 p-2 rounded text-xl font-mono z-10 flex flex-col gap-2">
          <span>Detecting: <span className="text-green-400 font-bold">{prediction}</span> ({confidence}%)</span>
          {!isRecording ? (
              <button onClick={handleRecordSubmit} className="bg-red-600 hover:bg-red-500 text-white font-bold py-1 px-4 rounded text-sm w-fit border-2 border-white shadow-xl shadow-red-900 animate-pulse">
                REC (Record 5-Second Sign)
              </button>
          ) : (
              <button onClick={handleStopRecording} className="bg-white hover:bg-gray-200 text-red-600 font-bold py-1 px-4 rounded text-sm w-fit border-2 border-red-600 shadow-xl shadow-red-900">
                STOP
              </button>
          )}
        </div>
      </div>

      <button onClick={downloadPDF} className="mt-8 bg-blue-600 hover:bg-blue-700 px-6 py-2 rounded font-bold">
        Download Session Transcript (PDF)
      </button>

      {transcript.length > 0 && (
         <div className="mt-6 p-4 bg-gray-800 rounded w-[640px]">
            <h3 className="font-bold text-gray-300 mb-2">Live Transcript:</h3>
            <p className="text-gray-400">{transcript.join(" ... ")}</p>
         </div>
      )}
    </div>
  );
}

export default App;