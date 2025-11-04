db = db.getSiblingDB('face_recog_db');

db.createUser({
  user: "face_user",
  pwd: "Face123!",
  roles: [
    { role: "readWrite", db: "face_recog_db" }
  ]
});

print("✅ User 'face_user' berhasil dibuat untuk database 'face_recog_db'.");
