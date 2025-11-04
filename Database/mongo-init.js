// mongo-init.js

db = db.getSiblingDB('face_recog_db'); // buat database utama

db.createUser({
  user: 'face_user',
  pwd: 'Face123!',
  roles: [
    {
      role: 'readWrite',
      db: 'face_recog_db'
    }
  ]
});

db.createCollection('detected_names');
print("✅ MongoDB initialized: Database 'face_recog_db' and user 'face_user' created successfully.");
