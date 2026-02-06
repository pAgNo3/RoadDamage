from flask import Flask, render_template, request, redirect, url_for, session, flash

app = Flask(__name__)
app.secret_key = 'frontend_key'

# --- FAKE DATABASE DATA ---
# (Your Senior will connect this to the real DB later)
MOCK_DATA = [
    {"lat": 28.6139, "lon": 77.2090, "type": "Detected", "desc": "Large Pothole"},
    {"lat": 28.6150, "lon": 77.2100, "type": "Predicted", "desc": "Cracks Forming"},
    {"lat": 28.6100, "lon": 77.2000, "type": "Detected", "desc": "Manhole Cover Missing"}
]

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form['username']
        pw = request.form['password']
        
        # Simple Check
        if user == 'admin' and pw == '1234':
            session['user'] = user
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid Credentials! Try admin / 1234')
            
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # We pass 'MOCK_DATA' to the frontend as a variable named 'potholes'
    return render_template('dashboard.html', user=session['user'], potholes=MOCK_DATA)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)