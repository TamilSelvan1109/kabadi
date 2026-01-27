#!/usr/bin/env python3
"""
Enhanced Skeleton Tracker Demo
Demonstrates the capabilities of the multi-player tracking system
"""

import cv2
import os
import sys
import time
from enhanced_skeleton_tracker import EnhancedSkeletonTracker

def check_requirements():
    """Check if all required files and dependencies are available"""
    print("Checking system requirements...")
    
    # Check MediaPipe model
    if not os.path.exists("pose_landmarker_lite.task"):
        print("⚠️  MediaPipe model not found. It will be downloaded automatically.")
    else:
        print("✅ MediaPipe model found")
    
    # Check video file
    video_files = ["assets/back_angle_video1.mp4", "assets/back_angle_video.MP4"]
    video_found = False
    for video_file in video_files:
        if os.path.exists(video_file):
            print(f"✅ Video file found: {video_file}")
            video_found = True
            break
    
    if not video_found:
        print("⚠️  No video file found in assets/ folder")
        print("   Please add a video file to test the system")
    
    # Check directories
    directories = ["violations", "violation_clips", "logs"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")
        else:
            print(f"✅ Directory exists: {directory}")
    
    return video_found

def run_demo():
    """Run the enhanced skeleton tracker demo"""
    print("\n" + "="*60)
    print("ENHANCED SKELETON TRACKER DEMO")
    print("="*60)
    print("Features Demonstration:")
    print("• Multi-player skeleton detection")
    print("• Red skeleton coloring for violations")
    print("• Automatic violation clip recording")
    print("• Real-time boundary detection")
    print("• Professional sports analytics")
    print("="*60)
    
    try:
        # Initialize tracker
        print("\nInitializing Enhanced Skeleton Tracker...")
        tracker = EnhancedSkeletonTracker()
        
        print("\nDemo Instructions:")
        print("1. Press SPACE to pause video")
        print("2. Press 'S' to enter selection mode")
        print("3. Click on player heads to assign IDs")
        print("4. Watch for red skeletons during violations")
        print("5. Check violation_clips/ folder for recorded clips")
        print("6. Press 'Q' to quit")
        
        input("\nPress Enter to start demo...")
        
        # Run the tracker
        tracker.run()
        
        print("\nDemo completed successfully!")
        print("Check the following folders for outputs:")
        print("• violations/ - Violation screenshots")
        print("• violation_clips/ - Violation video clips")
        print("• logs/ - System logs")
        
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        print("Please check the requirements and try again.")
        return False
    
    return True

def show_project_info():
    """Display project information"""
    print("\n" + "="*60)
    print("PROJECT INFORMATION")
    print("="*60)
    print("Title: Enhanced Multi-Player Skeleton Tracking System")
    print("Purpose: Final Year B.Tech Computer Vision Project")
    print("Domain: Sports Analytics & Computer Vision")
    print("Technology: MediaPipe, OpenCV, Python")
    print("="*60)
    
    print("\nKey Innovations:")
    print("• Advanced multi-player pose estimation")
    print("• Real-time violation detection and recording")
    print("• Color-coded skeleton visualization")
    print("• Automated evidence collection")
    print("• Professional sports officiating support")
    
    print("\nTechnical Highlights:")
    print("• 33-point skeleton tracking per player")
    print("• Confidence-based landmark filtering")
    print("• Temporal smoothing for accuracy")
    print("• Automatic video clip generation")
    print("• Configurable detection parameters")
    
    print("\nApplications:")
    print("• Professional Kabaddi match officiating")
    print("• Sports performance analysis")
    print("• Training and coaching tools")
    print("• Automated sports broadcasting")
    print("• Research in sports biomechanics")

def main():
    """Main demo function"""
    print("Enhanced Skeleton Tracker - Demo Mode")
    print("Final Year B.Tech Project Demonstration")
    print("-" * 50)
    
    # Show project information
    show_project_info()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Some requirements are missing.")
        print("Please ensure you have:")
        print("1. Video file in assets/ folder")
        print("2. All Python dependencies installed")
        print("3. Sufficient disk space for recordings")
        return
    
    # Ask user if they want to proceed
    print("\n" + "="*60)
    response = input("Ready to run demo? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        success = run_demo()
        if success:
            print("\n🎉 Demo completed successfully!")
            print("This system demonstrates advanced computer vision")
            print("capabilities suitable for a final year B.Tech project.")
        else:
            print("\n❌ Demo encountered issues.")
    else:
        print("\nDemo cancelled. You can run it anytime with:")
        print("python demo_enhanced.py")
    
    print("\nThank you for trying the Enhanced Skeleton Tracker!")

if __name__ == "__main__":
    main()