#!/usr/bin/env python3
"""
Kabadi Player Tracking System - Main Entry Point
Run this file to start the complete tracking system
"""

import os
import sys

def main():
    print("KABADI PLAYER TRACKING SYSTEM")
    print("=" * 50)
    print("Choose an option:")
    print("1. Set Boundary Lines (Line Detection)")
    print("2. Start Player Tracking")
    print("3. Exit")
    print("=" * 50)
    
    while True:
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            print("\nStarting Line Detection...")
            print("This will help you set boundary lines for violation detection")
            os.system("python line_detection.py")
            break
            
        elif choice == "2":
            print("\nStarting Player Tracking...")
            print("Make sure you have set boundary lines first (option 1)")
            os.system("python player_tracker.py")
            break
            
        elif choice == "3":
            print("\nGoodbye!")
            sys.exit(0)
            
        else:
            print("Invalid choice. Please enter 1-3.")

if __name__ == "__main__":
    main()