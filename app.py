from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import uuid
import time
from threading import Thread
from video_editor import build_final_video
import logging, traceback

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

jobs = {}

REQUIRED_SINGLE = ['hook', 'cta']
REQUIRED_MULTI_AT_LEAST_ONE = ['body']
ALLOWED_EXT = {
    'hook': {'.mp4', '.mov'},
    'body': {'.mp4', '.mov'},
    'cta': {'.mp4', '.mov'},
    'broll': {'.mp4', '.mov'},
    'overlay': {'.png', '.jpg', '.jpeg'},
    'music': {'.mp3', '.wav'}
}

def _ext_ok(field, filename):
    ext = os.path.splitext(filename.lower())[1]
    allowed = ALLOWED_EXT.get(field)
    return (allowed is None) or (ext in allowed)

def process_video_job(job_id, files, form):
    try:
        def report(pct: int, phase: str):
            jobs[job_id]['progress'] = pct
            jobs[job_id]['phase'] = phase

        jobs[job_id]['status'] = 'processing'
        report(5, "start")

        output_filename = build_final_video(job_id, UPLOAD_FOLDER, OUTPUT_FOLDER, form, report)

        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['video_url'] = f"/static/outputs/{output_filename}"
    except Exception as e:
        logging.exception(f"Job {job_id} failed")
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['error'] = f"{e.__class__.__name__}: {e}"
        jobs[job_id]['traceback'] = traceback.format_exc()

@app.route('/api/process', methods=['POST'])
def api_process():
    try:
        job_id = str(uuid.uuid4())
        jobs[job_id] = {'status': 'queued', 'progress': 0, 'video_url': None}

        # Basic validation
        # Check required single-file fields
        for field in REQUIRED_SINGLE:
            if field not in request.files or request.files[field].filename == '':
                jobs[job_id]['status'] = 'failed'
                jobs[job_id]['error'] = f"Missing required file: {field}"
                return jsonify({'error': f'Missing required file: {field}'}), 400

        # Check required multi-file groups
        for field in REQUIRED_MULTI_AT_LEAST_ONE:
            if not request.files.getlist(field):
                jobs[job_id]['status'] = 'failed'
                jobs[job_id]['error'] = f"At least one {field} file required"
                return jsonify({'error': f'At least one {field} file required'}), 400

        # Save all uploaded files
        for field in request.files:
            for file in request.files.getlist(field):
                if file.filename == '':
                    continue
                if not _ext_ok(field, file.filename):
                    jobs[job_id]['status'] = 'failed'
                    err = f"Invalid file type for {field}: {file.filename}"
                    jobs[job_id]['error'] = err
                    return jsonify({'error': err}), 400
                base, ext = os.path.splitext(file.filename)
                safe_name = f"{job_id}_{field}_{base}{ext}"
                target_path = os.path.join(UPLOAD_FOLDER, safe_name)
                counter = 1
                while os.path.exists(target_path):
                    safe_name = f"{job_id}_{field}_{base}_{counter}{ext}"
                    target_path = os.path.join(UPLOAD_FOLDER, safe_name)
                    counter += 1
                file.save(target_path)

        # Start processing thread
        thread = Thread(target=process_video_job, args=(job_id, request.files, request.form))
        thread.daemon = True
        thread.start()

        return jsonify({'job_id': job_id}), 202
    except Exception as e:
        logging.exception("Immediate /api/process failure")
        return jsonify({'error': f"{e.__class__.__name__}: {e}"}), 500

@app.route('/api/status/<job_id>')
def api_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'status': 'not_found'}), 404
    # Map status to step index
    if job['status'] == 'queued':
        current_step = 0
    elif job['status'] == 'processing':
        # Derive rough step from progress
        p = job.get('progress', 0)
        if p < 30:
            current_step = 1  # upload done -> processing
        elif p < 60:
            current_step = 2  # captions
        elif p < 90:
            current_step = 3  # insertions
        else:
            current_step = 4  # finalize
    elif job['status'] == 'completed':
        current_step = 4
    else:  # failed
        current_step = 4
    return jsonify({
        'status': job['status'],
        'progress': job.get('progress', 0),
        'current_step': current_step,
        'phase': job.get('phase'),
        'video_url': job.get('video_url'),
        'error': job.get('error'),
        'traceback': job.get('traceback')
    })

@app.route('/api/download/<filename>')
def api_download(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

@app.route('/static/outputs/<filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
