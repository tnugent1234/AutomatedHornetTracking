import cv2
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from datetime import datetime
from ultralytics import solutions
from matplotlib import cm
from matplotlib.colors import Normalize

# Create a timestamped subfolder for this analysis run
from datetime import datetime
analysis_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Results', analysis_timestamp)
os.makedirs(results_dir, exist_ok=True)

# Initialise counters
entries = 0
exits = 0
tracked_objects = {}  # To store track info: {'track_id': {'initial_pos': bool, 'current_pos': bool, 'active': bool}}
used_track_ids = set()  # To store IDs of all tracks that have ended

# Dictionary to store track durations and events
track_durations = {}  # {'track_id': {'start_frame': int, 'end_frame': int, 'entry': bool, 'exit': bool}}

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Hornet Tracker')
parser.add_argument('--video', type=str, help='Path to the video file for analysis', 
                    default="C://Users//tao213//Miniconda3//Test//MAH00001.1.mp4")
args = parser.parse_args()

# Get the source video name
video_path = args.video
video_name = os.path.basename(video_path)

with open(os.path.join(results_dir, 'Data.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Source Video', 'Timestamp (HH:MM:SS)', 'Event', 'Track ID', 'Total Entries', 'Total Exits', 'Active Tracks', 'Frame'])

def count_active_tracks():
    """Count the number of currently active tracks"""
    return sum(1 for track in tracked_objects.values() if track['active']) if tracked_objects else 0

def save_track_positions():
    """Save track position data to a CSV file"""
    if not hasattr(generate_heatmap, 'track_positions') or not generate_heatmap.track_positions:
        print("No track positions recorded to save.")
        return None
    
    csv_path = os.path.join(results_dir, 'Track Coordinates Data.csv')
    try:
        with open(os.path.join(results_dir, 'Results.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['X', 'Y', 'Frame'])
            for x, y, frame in generate_heatmap.track_positions:
                writer.writerow([x, y, frame])
        print(f"Track coordinates saved to: {csv_path}")
        return csv_path
    except Exception as e:
        print(f"Error saving track coordinates: {e}")
        return None

def save_duration_data():
    """Save track duration data to a CSV file"""
    duration_path = os.path.join(results_dir, 'Duration Data.csv')
    # Get the source video filename
    source_video = os.path.basename(video_path) if 'video_path' in globals() else 'Unknown'
    
    with open(duration_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Source Video', 'Track ID', 'Duration (Frames)', 'Duration (Seconds)', 'Entrance', 'Exit', 'First Frame', 'Last Frame'])
        
        # Get the total number of frames from the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for track_id, data in track_durations.items():
            if 'start_frame' in data:
                # Ensure we have valid frame numbers
                start_frame = max(0, data['start_frame'])
                end_frame = data.get('end_frame', total_frames - 1)
                
                # Ensure end_frame is at least start_frame (for 1-frame tracks)
                end_frame = max(end_frame, start_frame)
                # For tracks in both first and last frame, ensure duration is at least (total_frames - 1)
                if data.get('in_first_frame', False) and data.get('in_last_frame', False):
                    end_frame = max(end_frame, total_frames - 1)
                    
                duration_frames = end_frame - start_frame
                duration_seconds = duration_frames / generate_heatmap.fps if generate_heatmap.fps > 0 else 0
                
                writer.writerow([
                    source_video,
                    track_id,
                    duration_frames,  # Duration in frames
                    f"{duration_seconds:.2f}",  # Duration in seconds
                    'Yes' if data.get('entry', False) else 'No',
                    'Yes' if data.get('exit', False) else 'No',
                    'Yes' if data.get('in_first_frame', False) else 'No',
                    'Yes' if data.get('in_last_frame', False) else 'No'
                ])
    print(f"Duration data saved to: {duration_path}")
    return duration_path

# Create and write summary CSV header
with open(os.path.join(results_dir, 'Summary Results.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Source Video', 'Time (Minutes)', 'Entrances', 'Exits', 'Mean Active Tracks'])

# Dictionary to track minute-by-minute counts
minute_counts = {}

# Dictionary to track frame counts and active tracks per minute
minute_frames = {}

def update_summary(minute, is_entry):
    """Update the minute-by-minute counts"""
    if minute not in minute_counts:
        minute_counts[minute] = {'entries': 0, 'exits': 0, 'active_tracks_sum': 0, 'frame_count': 0}
    
    # Update entry/exit counts
    if is_entry:
        minute_counts[minute]['entries'] += 1
    else:
        minute_counts[minute]['exits'] += 1
    
    # Update active tracks sum and frame count for the current minute
    active_tracks = count_active_tracks()
    minute_counts[minute]['active_tracks_sum'] += active_tracks
    minute_counts[minute]['frame_count'] += 1
    
    # Write/update the summary file
    with open(os.path.join(results_dir, 'Summary Results.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Source Video', 'Minute', 'Entries', 'Exits', 'Mean Active Tracks'])
        for m in sorted(minute_counts.keys()):
            mean_active = minute_counts[m]['active_tracks_sum'] / minute_counts[m]['frame_count']
            writer.writerow([
                video_name,
                m, 
                minute_counts[m]['entries'], 
                minute_counts[m]['exits'], 
                f"{mean_active:.2f}"
            ])

def generate_heatmap(first_frame=None):
    """Generate two types of heatmaps: 
    1. Scatter plot of track positions over time
    2. Kernel density heatmap overlaid on first frame (if provided)
    """
    if not hasattr(generate_heatmap, 'track_positions') or not generate_heatmap.track_positions:
        print("No track positions recorded for heatmap generation.")
        return None
    
    try:
        # Extract coordinates and frame numbers
        x_coords = np.array([pos[0] for pos in generate_heatmap.track_positions])
        y_coords = np.array([pos[1] for pos in generate_heatmap.track_positions])
        frames = np.array([pos[2] for pos in generate_heatmap.track_positions])
        
        # 1. Scatter plot heatmap
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(x_coords, y_coords, c=frames, cmap='viridis', alpha=0.6, s=10)
        plt.colorbar(scatter, label='Frame Number')
        plt.title('Hornet Track Heatmap (Scatter)')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.gca().invert_yaxis()  # Invert y-axis to match video coordinate system
        plt.grid(True, alpha=0.3)
        scatter_path = os.path.join(results_dir, 'track_heatmap_scatter.png')
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Kernel density heatmap overlaid on first frame
        if first_frame is not None and len(x_coords) > 1:
            # Create a 2D histogram of the points
            heatmap, xedges, yedges = np.histogram2d(
                x_coords, 
                y_coords, 
                bins=50,  # Increased resolution
                range=[[0, first_frame.shape[1]], [0, first_frame.shape[0]]]
            )
            
            # Apply Gaussian filter for smoothing with increased intensity
            sigma = 1.5  # Adjust this value to control the amount of smoothing
            heatmap_smooth = gaussian_filter(heatmap, sigma=sigma)
            
            # Enhance contrast by applying power law (gamma) correction
            gamma = 0.7  # Values < 1 increase intensity of lower values
            heatmap_smooth = np.power(heatmap_smooth, gamma)
            
            # Normalise to 0-1 range for consistent colormapping
            heatmap_smooth = (heatmap_smooth - heatmap_smooth.min()) / (heatmap_smooth.max() - heatmap_smooth.min() + 1e-10)
            
            # Create figure with frame as background
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Display the original first frame (before any processing)
            if hasattr(generate_heatmap, 'original_first_frame'):
                display_frame = generate_heatmap.original_first_frame
            else:
                display_frame = first_frame
            # Display background frame with origin='upper' to match heatmap
            ax.imshow(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), origin='upper')
            
            # Display the smoothed heatmap with transparency
            extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]  # Regular y-coordinates (no flip)
            # Use viridis colormap with enhanced contrast
            cmap = plt.cm.viridis
            
            # Create the heatmap with increased contrast
            heatmap_display = ax.imshow(
                heatmap_smooth.T, 
                cmap=cmap,
                alpha=0.7,  # Slightly more opaque for better visibility
                extent=extent,
                aspect='equal',  # Maintain original aspect ratio
                origin='upper',  # Changed to 'upper' to match the flipped y-coordinates
                interpolation='bicubic',  # Smoother interpolation
                norm=Normalize(vmin=0, vmax=1, clip=True)  # Ensure consistent scaling
            )
            
            # Add colorbar
            cbar = plt.colorbar(heatmap_display, ax=ax, label='Density (smoothed)')
            
            # Add a title that indicates the smoothing
            ax.set_title('Hornet Track Density Heatmap'.format(sigma), pad=20)
            
            # Remove any existing annotations or ROI from the display
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            # Maintain aspect ratio without axis labels
            ax.set_aspect('equal')
            ax.axis('off')  # Turn off axis for cleaner visualisation
            
            # Save the density heatmap
            density_path = os.path.join(results_dir, 'track_density_heatmap.png')
            plt.savefig(density_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Scatter heatmap saved to: {scatter_path}")
            print(f"Density heatmap saved to: {density_path}")
            return scatter_path, density_path
        
        print(f"Scatter heatmap saved to: {scatter_path}")
        return scatter_path, None
        
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return None, None

# Initialise track positions list and frame rate
generate_heatmap.track_positions = []
generate_heatmap.fps = None
generate_heatmap.first_frame = None

def add_track_position(x, y, frame_num, frame=None):
    """Add a track position to the heatmap data
    
    Args:
        x, y: Position coordinates
        frame_num: Current frame number
        frame: Optional frame to store as the first frame for overlay
    """
    # Store the first frame if not already stored (before any processing)
    if frame is not None and not hasattr(generate_heatmap, 'original_first_frame'):
        # Store the very first frame before any processing
        generate_heatmap.original_first_frame = frame.copy()
        # Also store it as the first frame for backward compatibility
        generate_heatmap.first_frame = frame.copy()
    
    # Add position for every frame
    generate_heatmap.track_positions.append((x, y, frame_num))
    
    # Limit the number of points to prevent memory issues
    # Increased limit since we're sampling every frame
    if len(generate_heatmap.track_positions) > 100000:  # Keep last 100,000 points
        generate_heatmap.track_positions = generate_heatmap.track_positions[-100000:]

# Function to handle mouse events for circular region selection
def select_circle(event, x, y, flags, param):
    global center, radius, selecting
    
    global frame_size
    if event == cv2.EVENT_LBUTTONDOWN:
        center = (x, y)
        selecting = True
    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        radius = int(np.sqrt((x - center[0])**2 + (y - center[1])**2))
    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        # Store the frame size when circle is drawn
        if frame is not None:
            frame_size = (frame.shape[1], frame.shape[0])  # (width, height)
        
# Initialise global variables
center = None
radius = 0
selecting = False

# Initialise track durations dictionary
track_durations = {}

# Initialise tracked_objects dictionary
tracked_objects = {}
# Store the frame where the circle was drawn for reference
circle_draw_frame = None
frame_size = None  # Will store the size of the frame where circle was drawn

# Read video using the defined video_path
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), f"Error reading video file: {video_path}"

# Get video properties for sampling
generate_heatmap.fps = cap.get(cv2.CAP_PROP_FPS)
if generate_heatmap.fps <= 0:
    generate_heatmap.fps = 30  # Default to 30 FPS if not available

# Get first frame for region selection
ret, frame = cap.read()
if not ret:
    raise ValueError("Could not read first frame from video")

# Create window and set mouse callback
cv2.namedWindow("Select Circular Region")
cv2.setMouseCallback("Select Circular Region", select_circle)

# Let user select circular region
while True:
    img = frame.copy()
    
    # Draw the circle if we have a center and radius
    if center is not None and radius > 0:
        cv2.circle(img, center, radius, (0, 0, 255), 3)
    
    cv2.imshow("Select Circular Region", img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # Enter key
        if center is not None and radius > 0:
            break
    elif key == 27:  # ESC key
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow("Select Circular Region")

# Create points for circular region (approximated as a polygon with many points for smoothness)
num_points = 360  # Increased to 360 points for a very smooth circle
t = np.linspace(0, 2*np.pi, num_points, endpoint=False)  # endpoint=False to avoid duplicate point
region_points = [(int(center[0] + radius * np.cos(theta)), 
                 int(center[1] + radius * np.sin(theta))) 
                for theta in t]
# Ensure the polygon is closed by adding the first point at the end
region_points.append(region_points[0])

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
output_video_path = os.path.join(results_dir, 'Tracking Output.avi')
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialise object counter object without showing the default region
counter = solutions.ObjectCounter(
    show=True,  # display the output
    model="HornetTrackingModel.pt",  # model="yolo11n-obb.pt" for object counting with OBB model.
    # classes=[0, 2],  # count specific classes i.e. person and car with COCO pretrained model.
    tracker="customtrack.yaml",  # choose trackers i.e "bytetrack.yaml"
    conf=0.7,  # confidence threshold
)

# Initialise tracking variables
frame_count = 0
last_frame_objects = set()
current_frame_objects = set()  # Initialise current_frame_objects here

# Process first frame separately to initialise track_durations
success, im0 = cap.read()
if not success:
    raise ValueError("Could not read first frame from video")

# Get first frame detections
first_frame_results = counter.model.track(im0, persist=True, tracker="customtrack.yaml")
if first_frame_results[0].boxes is not None and first_frame_results[0].boxes.id is not None:
    first_frame_ids = first_frame_results[0].boxes.id.int().cpu().tolist()
    for track_id in first_frame_ids:
        if track_id not in track_durations:
            track_durations[track_id] = {
                'start_frame': 0,
                'entry': False,
                'exit': False,
                'in_first_frame': True,
                'in_last_frame': False
            }
        else:
            track_durations[track_id]['in_first_frame'] = True

# Reset video capture to beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Main processing loop
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing complete.")
        # At the end of the video, mark objects in the last frame and update their end_frame
    if not success and frame_count > 0:  # Only process if we've actually read frames
        last_frame = frame_count - 1  # frame_count is 1-based, frames are 0-based
        for track_id in current_frame_objects:
            if track_id in track_durations:
                track_durations[track_id]['in_last_frame'] = True
                # Update end_frame to the last frame if not already set
                if 'end_frame' not in track_durations[track_id] or track_durations[track_id]['end_frame'] < last_frame:
                    track_durations[track_id]['end_frame'] = last_frame
        break
    
    frame_count += 1
    frame = im0.copy()
    
    # Store the first frame for heatmap before any processing
    if frame_count == 1:
        generate_heatmap.original_first_frame = frame.copy()
    
    # Calculate relative position if frame size changed
    current_frame_size = (frame.shape[1], frame.shape[0])
    if frame_size is not None and current_frame_size != frame_size:
        # Calculate scaling factors
        scale_x = current_frame_size[0] / frame_size[0]
        scale_y = current_frame_size[1] / frame_size[1]
        # Scale center and radius
        scaled_center = (int(center[0] * scale_x), int(center[1] * scale_y))
        scaled_radius = int(radius * ((scale_x + scale_y) / 2))  # Average scale for radius
    else:
        scaled_center = center
        scaled_radius = radius
    
    # Draw the red circle outline with anti-aliasing and matching line thickness
    lw = max(round(sum(frame.shape) / 2 * 0.003), 2)  # Same as bounding box line width
    cv2.circle(frame, scaled_center, scaled_radius, (0, 0, 255), lw, lineType=cv2.LINE_AA)
    
    # Update the center and radius for tracking
    current_center = scaled_center
    current_radius = scaled_radius
    
    # Get detections with tracking
    results = counter.model.track(im0, persist=True, tracker="customtrack.yaml")
    
    # Store current frame objects for last frame check in next iteration
    last_frame_objects = current_frame_objects.copy()
    current_frame_objects = set()  # Reset for current frame
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        # current_frame_objects is already initialised above
        
        # Process each detection
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box[:4])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            # Use the current (possibly scaled) center and radius for distance calculation
            distance = np.sqrt((center_x - current_center[0])**2 + (center_y - current_center[1])**2)
            
            # Check if object is inside the circle
            is_inside = distance <= current_radius
            current_frame_objects.add(track_id)
            
            # Draw bounding box (Ultralytics YOLO style)
            color = (0, 0, 255)  # Colour for all boxes (BGR format)
            lw = max(round(sum(frame.shape) / 2 * 0.003), 2)  # Line width
            tf = max(lw - 1, 1)  # Font thickness
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=lw, lineType=cv2.LINE_AA)
            
            # Draw label background
            label = f"ID:{track_id}"
            w, h = cv2.getTextSize(label, 0, fontScale=lw/3, thickness=tf)[0]
            outside = y1 - h - 3 >= 0  # Label fits outside the box
            
            # Calculate background rectangle coordinates with better height matching
            if outside:
                bg_y1 = y1 - h - 3  # 3 pixels above text top
                bg_y2 = y1 + 3      # 3 pixels below text baseline
            else:
                bg_y1 = y2 + 3  # 3 pixels below the box
                bg_y2 = bg_y1 + h + 6  # Add extra padding (3px top + h + 3px bottom)
                
            cv2.rectangle(
                frame,
                (x1, bg_y1),
                (x1 + w, bg_y2),
                color,
                -1,
                cv2.LINE_AA
            )
            
            # Draw label text with proper vertical alignment
            if outside:
                text_y = y1 - 2  # Position for label above the box
            else:
                # For label below the box: y2 (bottom of box) + 3 (padding) + h (text height) + 2 (baseline offset)
                text_y = y2 + 3 + h + 2
                
            cv2.putText(
                frame,
                label,
                (x1, text_y),
                0,  # Font face (0 = default)
                lw / 3,  # Font scale
                (255, 255, 255),  # White text
                thickness=tf,

                lineType=cv2.LINE_AA
            )
            
            # Initialise new track - only if track_id doesn't exist, is not active, and hasn't been used before
            if (track_id not in tracked_objects or not tracked_objects[track_id]['active']) and track_id not in used_track_ids:
                # Initialise duration tracking if not already done
                if track_id not in track_durations:
                    track_durations[track_id] = {
                        'start_frame': frame_count,
                        'entry': False,
                        'exit': False,
                        'in_first_frame': False,
                        'in_last_frame': False
                    }
                # Store track info
                tracked_objects[track_id] = {
                    'initial_pos': is_inside,
                    'current_pos': is_inside,
                    'active': True,
                    'positions': [(int((x1 + x2) / 2), int((y1 + y2) / 2), is_inside, frame_count)],  # Store (x, y, is_inside, frame_count)
                    'entry_counted': False,
                    'exit_counted': False
                }
            else:
                # Update position and store for heatmap
                tracked_objects[track_id]['current_pos'] = is_inside
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)
                # Store (x, y, is_inside, frame_count) for each position
                tracked_objects[track_id]['positions'].append((x_center, y_center, is_inside, frame_count))
                add_track_position(x_center, y_center, frame_count, frame)
                tracked_objects[track_id]['active'] = True
    
    # Check for completed tracks and count entries/exits
    for track_id in list(tracked_objects.keys()):
        if track_id not in current_frame_objects and tracked_objects[track_id]['active']:
            # This track has ended
            track = tracked_objects[track_id]
            
            # Get first and last positions from track history
            if track['positions']:  # Ensure there are positions recorded
                first_pos = track['positions'][0][2]  # [x, y, is_inside] for first position
                last_pos = track['positions'][-1][2]   # [x, y, is_inside] for last position
                
                # Calculate timestamp based on video position (HH:MM:SS format)
                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                hours = timestamp_ms // 3600000
                minutes = (timestamp_ms % 3600000) // 60000
                seconds = (timestamp_ms % 60000) // 1000
                current_minute = f"{hours:02d}:{minutes:02d}"
                timestamp_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                # Only check entry/exit when the track ends (not active anymore)
                if not track['entry_counted'] or not track['exit_counted']:
                    active_tracks = count_active_tracks()
                    
                    # Check for entry (started outside, ended inside)
                    if not first_pos and last_pos and not track['entry_counted']:
                        entries += 1
                        track['entry_counted'] = True
                        if track_id in track_durations:
                            track_durations[track_id]['entry'] = True
                        
                        with open(os.path.join(results_dir, 'Data.csv'), 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([video_name, timestamp_str, 'ENTRY', track_id, entries, exits, active_tracks, frame_count])
                        print(f"Object {track_id} completed entry at {timestamp_str}. Total entries: {entries}, Active tracks: {active_tracks}")
                        update_summary(current_minute, is_entry=True)
                    
                    # Check for exit (started inside, ended outside)
                    elif first_pos and not last_pos and not track['exit_counted']:
                        exits += 1
                        track['exit_counted'] = True
                        if track_id in track_durations:
                            track_durations[track_id]['exit'] = True
                        
                        with open(os.path.join(results_dir, 'Data.csv'), 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([video_name, timestamp_str, 'EXIT', track_id, entries, exits, active_tracks, frame_count])
                        print(f"Object {track_id} completed exit at {timestamp_str}. Total exits: {exits}, Active tracks: {active_tracks}")
                        update_summary(current_minute, is_entry=False)
            
            # Update end frame, mark as inactive, and add to used_track_ids
            if track_id in track_durations:
                track_durations[track_id]['end_frame'] = frame_count
            tracked_objects[track_id]['active'] = False
            used_track_ids.add(track_id)
    
    # Only remove tracks that are no longer active and have been processed
    # Keep all tracks in tracked_objects for data collection
    for track_id in list(tracked_objects.keys()):
        if track_id not in current_frame_objects:
            tracked_objects[track_id]['active'] = False
    
    # Display counts on frame
    cv2.putText(frame, f'Entries: {entries}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Exits: {exits}', (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Tracks: {len([x for x in tracked_objects.values() if x])}', (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Write the frame
    video_writer.write(frame)
    
    # Show frame (optional, for debugging)
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()

# Generate heatmaps before closing, using the original unprocessed first frame
scatter_path, density_path = generate_heatmap(generate_heatmap.original_first_frame)
if scatter_path:
    print(f"Scatter heatmap saved to: {scatter_path}")
if density_path:
    print(f"Density heatmap saved to: {density_path}")

# Save duration data
duration_path = save_duration_data()

# Save track positions to CSV
def save_track_positions():
    # Get the source video filename
    source_video = os.path.basename(video_path) if 'video_path' in globals() else 'Unknown'
    
    track_positions_path = os.path.join(results_dir, 'Track Coordinates Data.csv')
    with open(track_positions_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Source Video', 'Track ID', 'X', 'Y', 'Frame'])
        # Sort tracks by their first appearance to maintain consistent ordering
        sorted_tracks = sorted(tracked_objects.items(), key=lambda x: x[1]['positions'][0][3] if x[1]['positions'] else float('inf'))
        for track_id, track in sorted_tracks:
            if 'positions' in track:  # Ensure the track has position data
                for position in track['positions']:
                    if len(position) >= 4:  # Ensure position has all required elements
                        writer.writerow([source_video, track_id, position[0], position[1], position[3]])
    return track_positions_path

track_coords_path = save_track_positions()

def generate_track_visualisation(first_frame, tracked_objects):
    """
    Generate a visualisation of all tracks overlaid on the first video frame.
    Each track is shown as a colored line with a unique color per track ID.
    
    Args:
        first_frame: The first frame of the video
        tracked_objects: Dictionary containing all tracked objects and their positions
    """
    if first_frame is None:
        print("Error: First frame is not available for track visualisation")
        return None
    
    # Create a copy of the first frame to draw on
    vis_frame = first_frame.copy()
    
    # Create a colormap for unique track colors
    cmap = plt.get_cmap('tab20')
    num_tracks = len(tracked_objects)
    
    # Draw each track
    for i, (track_id, track) in enumerate(tracked_objects.items()):
        if 'positions' not in track or len(track['positions']) < 2:
            continue
            
        # Get a unique color for this track
        color = np.array(cmap(i % cmap.N)) * 255  # Convert to 0-255 range for OpenCV
        color = tuple(map(int, color[:3]))  # Convert to BGR for OpenCV
        
        # Draw the track as a series of connected lines
        points = np.array([(p[0], p[1]) for p in track['positions']], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(vis_frame, [points], isClosed=False, color=color, thickness=2)
        
        # Add track ID label at the start of the track
        if len(track['positions']) > 0:
            start_point = (int(track['positions'][0][0]), int(track['positions'][0][1]))
            cv2.putText(vis_frame, str(track_id), 
                       (start_point[0] + 5, start_point[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save the visualisation
    output_path = os.path.join(results_dir, 'track_visualisation.png')
    cv2.imwrite(output_path, vis_frame)
    print(f"Track visualisation saved to: {output_path}")
    return output_path

# Generate the track visualisation if we have a first frame
if generate_heatmap.original_first_frame is not None:
    vis_path = generate_track_visualisation(generate_heatmap.original_first_frame, tracked_objects)

cv2.destroyAllWindows()  # destroy all opened windows