# Collaborator Access - Auto Video Editor Backend

## Overview
The Auto Video Editor backend now includes a collaborator dashboard that provides administrative access for monitoring backend operations, system status, and job processing.

## Accessing the Collaborator Dashboard

### URL
Navigate to: `http://your-server:port/collaborator`

**Examples:**
- Local development: `http://localhost:5000/collaborator`  
- Production: `https://your-domain.com/collaborator`

### Features

The collaborator dashboard provides:

1. **System Monitoring**
   - CPU usage in real-time
   - Memory consumption
   - Disk usage
   - System platform information

2. **Job Statistics**
   - Total number of jobs processed
   - Current queue status (queued, processing, completed, failed)
   - Real-time job counters

3. **Recent Job History**
   - Last 10 jobs with their status
   - Progress indicators
   - Error messages for failed jobs
   - Processing phases

4. **Backend Information**
   - Python version
   - Available API endpoints
   - Application framework details

### Auto-Refresh
The dashboard automatically refreshes every 30 seconds to provide up-to-date information.

### Navigation
- **Main App** button: Returns to the main video editing interface
- **Refresh** button: Manually refresh the dashboard

## API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main video editing interface |
| `/collaborator` | GET | Administrative dashboard |
| `/api/process` | POST | Start video processing job |
| `/api/status/<job_id>` | GET | Check job status |
| `/api/download/<filename>` | GET | Download processed video |

## Security Note
The collaborator dashboard currently has no authentication. In production environments, consider implementing proper access controls or restricting access through network-level security measures.

## Technical Details
- Built with Flask framework
- Uses psutil for system monitoring
- Responsive design with Tailwind CSS
- Real-time status updates