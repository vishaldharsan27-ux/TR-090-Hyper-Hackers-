import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyABsdYTLR2WnWRrbJKBDiRfNX6CufZOtoA",
  authDomain: "isl-translator-9c531.firebaseapp.com",
  projectId: "isl-translator-9c531",
  storageBucket: "isl-translator-9c531.firebasestorage.app",
  messagingSenderId: "756438953839",
  appId: "1:756438953839:web:d402c24881641d5da6829a"
};

const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);