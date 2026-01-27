import os
import json
import time
from datetime import datetime, timezone
import uuid
import base64
from typing import Dict, List, Optional
import threading
import queue

# Cloud Database Imports
try:
    import pymongo
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("Warning: pymongo not installed. MongoDB features disabled.")

try:
    import firebase_admin
    from firebase_admin import credentials, firestore, storage
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("Warning: firebase-admin not installed. Firebase features disabled.")

class CloudViolationLogger:
    def __init__(self, config_file="cloud_config.json"):
        self.config = self.load_config(config_file)
        self.mongodb_client = None
        self.firebase_db = None
        self.firebase_bucket = None
        self.log_queue = queue.Queue()
        self.is_running = True
        
        # Initialize cloud connections
        self.init_mongodb()
        self.init_firebase()
        
        # Start background logging thread
        self.logging_thread = threading.Thread(target=self._background_logger, daemon=True)
        self.logging_thread.start()
        
        print("🌐 Cloud Violation Logger initialized")
    
    def load_config(self, config_file):
        """Load cloud configuration"""
        default_config = {
            "mongodb": {
                "enabled": True,
                "connection_string": "mongodb+srv://username:password@cluster.mongodb.net/",
                "database": "kabaddi_tracking",
                "collection": "violations"
            },
            "firebase": {
                "enabled": True,
                "service_account_key": "firebase-service-account.json",
                "storage_bucket": "kabaddi-tracking.appspot.com",
                "collection": "violations"
            },
            "local_backup": {
                "enabled": True,
                "directory": "violation_logs",
                "max_files": 100
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key not in config:
                        config[key] = default_config[key]
                return config
        except FileNotFoundError:
            # Create default config file
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            print(f"Created default config file: {config_file}")
            return default_config
    
    def init_mongodb(self):
        """Initialize MongoDB Atlas connection"""
        if not MONGODB_AVAILABLE or not self.config["mongodb"]["enabled"]:
            return
        
        try:
            self.mongodb_client = MongoClient(
                self.config["mongodb"]["connection_string"],
                serverSelectionTimeoutMS=5000
            )
            # Test connection
            self.mongodb_client.server_info()
            self.mongodb_db = self.mongodb_client[self.config["mongodb"]["database"]]
            self.mongodb_collection = self.mongodb_db[self.config["mongodb"]["collection"]]
            print("✅ MongoDB Atlas connected successfully")
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            self.mongodb_client = None
    
    def init_firebase(self):
        """Initialize Firebase connection"""
        if not FIREBASE_AVAILABLE or not self.config["firebase"]["enabled"]:
            return
        
        try:
            # Initialize Firebase Admin SDK
            if not firebase_admin._apps:
                cred = credentials.Certificate(self.config["firebase"]["service_account_key"])
                firebase_admin.initialize_app(cred, {
                    'storageBucket': self.config["firebase"]["storage_bucket"]
                })
            
            self.firebase_db = firestore.client()
            self.firebase_bucket = storage.bucket()
            print("✅ Firebase connected successfully")
        except Exception as e:
            print(f"❌ Firebase connection failed: {e}")
            self.firebase_db = None
    
    def log_violation(self, player_id: str, frame_number: int, coordinates: Dict, 
                     screenshot: Optional[bytes] = None, match_context: Dict = None):
        """Log a violation to cloud databases"""
        violation_entry = self.create_violation_entry(
            player_id, frame_number, coordinates, screenshot, match_context
        )
        
        # Add to queue for background processing
        self.log_queue.put(violation_entry)
        
        # Also save locally immediately
        self.save_local_backup(violation_entry)
        
        return violation_entry["violation_id"]
    
    def create_violation_entry(self, player_id: str, frame_number: int, 
                             coordinates: Dict, screenshot: Optional[bytes] = None,
                             match_context: Dict = None) -> Dict:
        """Create a structured violation entry"""
        now = datetime.now(timezone.utc)
        violation_id = f"VID_{now.strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        entry = {
            "violation_id": violation_id,
            "match_id": match_context.get("match_id", "MATCH_DEFAULT") if match_context else "MATCH_DEFAULT",
            "player_id": player_id,
            "player_name": match_context.get("player_name", f"Player_{player_id}") if match_context else f"Player_{player_id}",
            "team": match_context.get("team", "Unknown") if match_context else "Unknown",
            "violation_type": "BOUNDARY_CROSSING",
            "timestamp": now.isoformat(),
            "frame_number": frame_number,
            "video_timestamp": self.frame_to_timestamp(frame_number),
            "coordinates": coordinates,
            "confidence_score": coordinates.get("confidence", 0.0),
            "evidence": {
                "screenshot_captured": screenshot is not None,
                "screenshot_size": len(screenshot) if screenshot else 0
            },
            "referee_review": {
                "status": "PENDING",
                "reviewed_by": None,
                "review_timestamp": None,
                "notes": ""
            },
            "match_context": match_context or {},
            "system_info": {
                "detection_method": "MediaPipe_33_Landmark",
                "processing_time": time.time(),
                "system_version": "1.0.0"
            }
        }
        
        # Add screenshot to evidence if provided
        if screenshot:
            screenshot_filename = f"{violation_id}_screenshot.jpg"
            entry["evidence"]["screenshot_filename"] = screenshot_filename
            entry["evidence"]["screenshot_data"] = base64.b64encode(screenshot).decode('utf-8')
        
        return entry
    
    def frame_to_timestamp(self, frame_number: int, fps: int = 30) -> str:
        """Convert frame number to video timestamp"""
        total_seconds = frame_number / fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    
    def _background_logger(self):
        """Background thread for cloud logging"""
        while self.is_running:
            try:
                # Get violation from queue (timeout after 1 second)
                violation = self.log_queue.get(timeout=1)
                
                # Log to MongoDB
                if self.mongodb_client:
                    self.log_to_mongodb(violation)
                
                # Log to Firebase
                if self.firebase_db:
                    self.log_to_firebase(violation)
                
                self.log_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Background logging error: {e}")
    
    def log_to_mongodb(self, violation: Dict):
        """Log violation to MongoDB Atlas"""
        try:
            # Remove screenshot data for MongoDB (too large)
            mongodb_entry = violation.copy()
            if "screenshot_data" in mongodb_entry["evidence"]:
                del mongodb_entry["evidence"]["screenshot_data"]
            
            result = self.mongodb_collection.insert_one(mongodb_entry)
            print(f"✅ MongoDB: Logged {violation['violation_id']}")
            return result.inserted_id
        except Exception as e:
            print(f"❌ MongoDB logging error: {e}")
            return None
    
    def log_to_firebase(self, violation: Dict):
        """Log violation to Firebase Firestore"""
        try:
            # Upload screenshot to Firebase Storage if available
            if "screenshot_data" in violation["evidence"]:
                screenshot_url = self.upload_screenshot_to_firebase(
                    violation["violation_id"], 
                    violation["evidence"]["screenshot_data"]
                )
                violation["evidence"]["screenshot_url"] = screenshot_url
                del violation["evidence"]["screenshot_data"]  # Remove base64 data
            
            # Add to Firestore
            doc_ref = self.firebase_db.collection(self.config["firebase"]["collection"]).document(violation["violation_id"])
            doc_ref.set(violation)
            print(f"✅ Firebase: Logged {violation['violation_id']}")
            return doc_ref.id
        except Exception as e:
            print(f"❌ Firebase logging error: {e}")
            return None
    
    def upload_screenshot_to_firebase(self, violation_id: str, screenshot_data: str) -> str:
        """Upload screenshot to Firebase Storage"""
        try:
            # Decode base64 screenshot
            screenshot_bytes = base64.b64decode(screenshot_data)
            
            # Upload to Firebase Storage
            blob = self.firebase_bucket.blob(f"violations/{violation_id}_screenshot.jpg")
            blob.upload_from_string(screenshot_bytes, content_type='image/jpeg')
            
            # Make public and return URL
            blob.make_public()
            return blob.public_url
        except Exception as e:
            print(f"❌ Screenshot upload error: {e}")
            return ""
    
    def save_local_backup(self, violation: Dict):
        """Save violation to local backup"""
        if not self.config["local_backup"]["enabled"]:
            return
        
        try:
            backup_dir = self.config["local_backup"]["directory"]
            os.makedirs(backup_dir, exist_ok=True)
            
            # Save violation JSON
            filename = f"{violation['violation_id']}.json"
            filepath = os.path.join(backup_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(violation, f, indent=2)
            
            # Save screenshot separately if available
            if "screenshot_data" in violation["evidence"]:
                screenshot_filename = f"{violation['violation_id']}_screenshot.jpg"
                screenshot_path = os.path.join(backup_dir, screenshot_filename)
                
                screenshot_bytes = base64.b64decode(violation["evidence"]["screenshot_data"])
                with open(screenshot_path, 'wb') as f:
                    f.write(screenshot_bytes)
            
            # Cleanup old files if needed
            self.cleanup_old_backups()
            
        except Exception as e:
            print(f"❌ Local backup error: {e}")
    
    def cleanup_old_backups(self):
        """Remove old backup files to maintain limit"""
        try:
            backup_dir = self.config["local_backup"]["directory"]
            max_files = self.config["local_backup"]["max_files"]
            
            # Get all JSON files (violation logs)
            json_files = [f for f in os.listdir(backup_dir) if f.endswith('.json')]
            
            if len(json_files) > max_files:
                # Sort by creation time and remove oldest
                json_files.sort(key=lambda x: os.path.getctime(os.path.join(backup_dir, x)))
                files_to_remove = json_files[:-max_files]
                
                for filename in files_to_remove:
                    os.remove(os.path.join(backup_dir, filename))
                    # Also remove corresponding screenshot
                    screenshot_file = filename.replace('.json', '_screenshot.jpg')
                    screenshot_path = os.path.join(backup_dir, screenshot_file)
                    if os.path.exists(screenshot_path):
                        os.remove(screenshot_path)
                
                print(f"🧹 Cleaned up {len(files_to_remove)} old backup files")
        except Exception as e:
            print(f"❌ Cleanup error: {e}")
    
    def get_violations(self, match_id: str = None, player_id: str = None, 
                      limit: int = 100) -> List[Dict]:
        """Retrieve violations from cloud database"""
        violations = []
        
        # Try MongoDB first
        if self.mongodb_client:
            try:
                query = {}
                if match_id:
                    query["match_id"] = match_id
                if player_id:
                    query["player_id"] = player_id
                
                cursor = self.mongodb_collection.find(query).limit(limit).sort("timestamp", -1)
                violations = list(cursor)
                
                # Convert ObjectId to string for JSON serialization
                for violation in violations:
                    violation["_id"] = str(violation["_id"])
                
                return violations
            except Exception as e:
                print(f"❌ MongoDB query error: {e}")
        
        # Fallback to Firebase
        if self.firebase_db and not violations:
            try:
                query = self.firebase_db.collection(self.config["firebase"]["collection"])
                
                if match_id:
                    query = query.where("match_id", "==", match_id)
                if player_id:
                    query = query.where("player_id", "==", player_id)
                
                docs = query.limit(limit).order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
                violations = [doc.to_dict() for doc in docs]
                
                return violations
            except Exception as e:
                print(f"❌ Firebase query error: {e}")
        
        return violations
    
    def get_match_statistics(self, match_id: str) -> Dict:
        """Get comprehensive match statistics"""
        violations = self.get_violations(match_id=match_id)
        
        stats = {
            "match_id": match_id,
            "total_violations": len(violations),
            "players_violated": len(set(v["player_id"] for v in violations)),
            "violation_timeline": [],
            "player_stats": {},
            "team_stats": {}
        }
        
        # Process violations for statistics
        for violation in violations:
            player_id = violation["player_id"]
            team = violation.get("team", "Unknown")
            
            # Player statistics
            if player_id not in stats["player_stats"]:
                stats["player_stats"][player_id] = {
                    "violations": 0,
                    "first_violation": violation["timestamp"],
                    "last_violation": violation["timestamp"],
                    "team": team
                }
            
            stats["player_stats"][player_id]["violations"] += 1
            stats["player_stats"][player_id]["last_violation"] = violation["timestamp"]
            
            # Team statistics
            if team not in stats["team_stats"]:
                stats["team_stats"][team] = {"violations": 0, "players_violated": set()}
            
            stats["team_stats"][team]["violations"] += 1
            stats["team_stats"][team]["players_violated"].add(player_id)
            
            # Timeline
            stats["violation_timeline"].append({
                "timestamp": violation["timestamp"],
                "player_id": player_id,
                "team": team,
                "frame": violation["frame_number"]
            })
        
        # Convert sets to counts for JSON serialization
        for team in stats["team_stats"]:
            stats["team_stats"][team]["players_violated"] = len(stats["team_stats"][team]["players_violated"])
        
        return stats
    
    def close(self):
        """Close cloud connections"""
        self.is_running = False
        
        if self.logging_thread.is_alive():
            self.logging_thread.join(timeout=5)
        
        if self.mongodb_client:
            self.mongodb_client.close()
        
        print("🔒 Cloud Violation Logger closed")

# Global instance
cloud_logger = None

def init_cloud_logger():
    """Initialize global cloud logger"""
    global cloud_logger
    if cloud_logger is None:
        cloud_logger = CloudViolationLogger()
    return cloud_logger

def log_violation_to_cloud(player_id: str, frame_number: int, coordinates: Dict,
                          screenshot: Optional[bytes] = None, match_context: Dict = None):
    """Convenience function to log violation"""
    logger = init_cloud_logger()
    return logger.log_violation(player_id, frame_number, coordinates, screenshot, match_context)