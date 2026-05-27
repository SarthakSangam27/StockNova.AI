import json
import math
import random
import os
import re
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
import bcrypt
import jwt

# ============================================================
# CONFIGURATION
# ============================================================

app = Flask(__name__)
CORS(app)

# Secret keys - change these in production!
SECRET_KEY = os.environ.get('SECRET_KEY', 'your-super-secret-key-change-in-production')
JWT_SECRET = os.environ.get('JWT_SECRET', 'your-jwt-secret-key-change-in-production')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = 24

# Database file
DB_FILE = 'users.json'

# ============================================================
# STOCK TICKER PARAMETERS (existing code)
# ============================================================

TICKER_PARAMS = {
    "AAPL": (178, 0.00042, 0.017, 5e7, "Apple Inc.", "Technology"),
    "TSLA": (235, 0.0005,  0.031, 8e7, "Tesla Inc.", "Automotive"),
    "NVDA": (480, 0.0008,  0.024, 4e7, "NVIDIA Corp.", "Semiconductors"),
    "MSFT": (370, 0.0004,  0.015, 2e7, "Microsoft Corp.", "Technology"),
    "AMZN": (175, 0.0003,  0.018, 3e7, "Amazon.com", "Consumer"),
    "GOOGL":(140, 0.00035, 0.016, 2e7, "Alphabet Inc.", "Technology"),
    "META": (320, 0.0005,  0.021,1.5e7,"Meta Platforms","Social Media"),
    "AMD":  (120, 0.0006,  0.026, 5e7, "Advanced Micro Devices","Semiconductors"),
}

_cache = {}

# ============================================================
# USER DATABASE FUNCTIONS
# ============================================================

def load_users():
    """Load users from JSON file"""
    if not os.path.exists(DB_FILE):
        return {}
    try:
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def save_users(users):
    """Save users to JSON file"""
    with open(DB_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def user_exists(username, email):
    """Check if username or email already exists"""
    users = load_users()
    for uid, user_data in users.items():
        if user_data.get('username', '').lower() == username.lower():
            return True, 'username'
        if user_data.get('email', '').lower() == email.lower():
            return True, 'email'
    return False, None

def create_user(username, email, password, role='user'):
    """Create a new user with hashed password"""
    users = load_users()
    
    # Check if user exists
    exists, field = user_exists(username, email)
    if exists:
        return None, f"{field.capitalize()} already exists"
    
    # Hash password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    
    # Create user data
    user_id = f"user_{len(users) + 1}_{int(datetime.now().timestamp())}"
    now = datetime.now().isoformat()
    
    users[user_id] = {
        'id': user_id,
        'username': username,
        'email': email,
        'password': hashed.decode('utf-8'),
        'role': role,
        'created_at': now,
        'updated_at': now,
        'is_active': True,
        'profile': {
            'first_name': '',
            'last_name': '',
            'bio': '',
            'avatar_url': '',
            'preferences': {
                'theme': 'light',
                'notifications': True
            }
        }
    }
    
    save_users(users)
    return users[user_id], None

def get_user_by_username(username):
    """Get user by username"""
    users = load_users()
    for uid, user_data in users.items():
        if user_data.get('username', '').lower() == username.lower():
            return user_data
    return None

def get_user_by_id(user_id):
    """Get user by ID"""
    users = load_users()
    return users.get(user_id)

def verify_password(user_data, password):
    """Verify password against stored hash"""
    try:
        stored_hash = user_data.get('password', '').encode('utf-8')
        return bcrypt.checkpw(password.encode('utf-8'), stored_hash)
    except Exception:
        return False

def update_user(user_id, updates):
    """Update user data"""
    users = load_users()
    if user_id not in users:
        return None, "User not found"
    
    # Don't allow updating these fields directly
    protected = ['id', 'password', 'role', 'created_at']
    for field in protected:
        if field in updates:
            del updates[field]
    
    updates['updated_at'] = datetime.now().isoformat()
    
    # Handle profile updates
    if 'profile' in updates:
        current_profile = users[user_id].get('profile', {})
        users[user_id]['profile'] = {**current_profile, **updates['profile']}
        del updates['profile']
    
    users[user_id].update(updates)
    save_users(users)
    return users[user_id], None

def delete_user(user_id):
    """Delete a user"""
    users = load_users()
    if user_id not in users:
        return False, "User not found"
    
    del users[user_id]
    save_users(users)
    return True, None

# ============================================================
# JWT TOKEN FUNCTIONS
# ============================================================

def generate_token(user_data):
    """Generate JWT token for user"""
    payload = {
        'user_id': user_data['id'],
        'username': user_data['username'],
        'email': user_data['email'],
        'role': user_data.get('role', 'user'),
        'exp': datetime.now() + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.now()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(token):
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload, None
    except jwt.ExpiredSignatureError:
        return None, "Token has expired"
    except jwt.InvalidTokenError as e:
        return None, f"Invalid token: {str(e)}"

def refresh_token(old_token):
    """Refresh an existing token"""
    payload, error = verify_token(old_token)
    if error:
        return None, error
    
    user_data = get_user_by_id(payload['user_id'])
    if not user_data:
        return None, "User not found"
    
    return generate_token(user_data), None

# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not re.search(r'[A-Za-z]', password):
        return False, "Password must contain at least one letter"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    return True, None

def validate_username(username):
    """Validate username format"""
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    if len(username) > 30:
        return False, "Username must be at most 30 characters"
    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        return False, "Username can only contain letters, numbers, underscores, and hyphens"
    return True, None

def validate_registration_data(data):
    """Validate all registration data"""
    errors = {}
    
    # Username validation
    valid, msg = validate_username(data.get('username', ''))
    if not valid:
        errors['username'] = msg
    
    # Email validation
    email = data.get('email', '')
    if not email:
        errors['email'] = "Email is required"
    elif not validate_email(email):
        errors['email'] = "Invalid email format"
    
    # Password validation
    valid, msg = validate_password(data.get('password', ''))
    if not valid:
        errors['password'] = msg
    
    # Confirm password
    if data.get('password') != data.get('confirm_password'):
        errors['confirm_password'] = "Passwords do not match"
    
    return errors

# ============================================================
# AUTH DECORATORS
# ============================================================

def token_required(f):
    """Decorator to require valid JWT token"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Get token from header
        auth_header = request.headers.get('Authorization')
        if auth_header:
            try:
                token = auth_header.split(' ')[1]
            except IndexError:
                return jsonify({'error': 'Invalid authorization header format'}), 401
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        # Verify token
        payload, error = verify_token(token)
        if error:
            return jsonify({'error': error}), 401
        
        # Add user to request context
        request.user = payload
        return f(*args, **kwargs)
    
    return decorated

def role_required(*allowed_roles):
    """Decorator to require specific role(s)"""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            user_role = request.user.get('role', 'user')
            if user_role not in allowed_roles:
                return jsonify({'error': 'Insufficient permissions'}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator

def admin_required(f):
    """Decorator to require admin role"""
    return role_required('admin')(f)

# ============================================================
# STOCK DATA FUNCTIONS (existing code)
# ============================================================

def generate_data(ticker):
    """Generate realistic yfinance-compatible OHLCV data using GBM"""
    if ticker in _cache:
        return _cache[ticker]
    p0, drift, sigma, vol_base, name, sector = TICKER_PARAMS.get(
        ticker, (100, 0.0003, 0.02, 2e7, ticker, "Unknown"))
    rows, price = [], p0
    from datetime import date, timedelta
    d = date(2022, 1, 3)
    spare = [None]
    def bm():
        if spare[0] is not None:
            t, spare[0] = spare[0], None; return t
        u,v,s=0,0,0
        while s>=1 or s==0:
            u=random.random()*2-1; v=random.random()*2-1; s=u*u+v*v
        m=math.sqrt(-2*math.log(s)/s); spare[0]=v*m; return u*m
    random.seed(sum(ord(c) for c in ticker))
    for _ in range(600):
        while d.weekday()>=5: d+=timedelta(1)
        rnd=bm()
        price=price*math.exp(drift+sigma*rnd)
        rng=price*(0.008+abs(rnd)*0.005)
        o=price*(1+(random.random()-0.5)*0.006)
        h=max(o,price)+rng*random.random()
        l=min(o,price)-rng*random.random()
        rows.append({
            "date":str(d),"open":round(o,2),"high":round(h,2),
            "low":round(l,2),"close":round(price,2),
            "volume":int(vol_base*(0.4+random.random()*1.2))
        })
        d+=timedelta(1)
    _cache[ticker] = rows
    return rows

def ema(vals, period):
    if not vals: return []
    k=2/(period+1); r=[vals[0]]
    for v in vals[1:]: r.append(v*k+r[-1]*(1-k))
    return r

def sma(vals, period):
    r=[None]*(period-1)
    for i in range(period-1, len(vals)):
        r.append(sum(vals[i-period+1:i+1])/period)
    return r

def rsi_fn(closes, period=14):
    if len(closes)<=period: return [None]*len(closes)
    gains,losses=[],[]
    for i in range(1,len(closes)):
        d=closes[i]-closes[i-1]
        gains.append(max(d,0)); losses.append(max(-d,0))
    out=[None]*period
    ag=sum(gains[:period])/period; al=sum(losses[:period])/period
    out.append(100-(100/(1+ag/al)) if al else 100)
    for i in range(period,len(gains)):
        ag=(ag*(period-1)+gains[i])/period
        al=(al*(period-1)+losses[i])/period
        out.append(100-(100/(1+ag/al)) if al else 100)
    return out

def macd_fn(closes):
    e12=ema(closes,12); e26=ema(closes,26)
    m=[a-b for a,b in zip(e12,e26)]
    sig=ema(m,9); hist=[a-b for a,b in zip(m,sig)]
    return m,sig,hist

def bollinger_fn(closes,period=20):
    sm=sma(closes,period); out=[]
    for i in range(len(closes)):
        if sm[i] is None: out.append((None,None,None)); continue
        slc=closes[max(0,i-period+1):i+1]
        std=math.sqrt(sum((x-sm[i])**2 for x in slc)/len(slc))
        out.append((round(sm[i]+2*std,2),round(sm[i],2),round(sm[i]-2*std,2)))
    return out

def compute_indicators(rows):
    closes=[r["close"] for r in rows]
    highs=[r["high"] for r in rows]; lows=[r["low"] for r in rows]
    e20=ema(closes,20); e50=ema(closes,50); e200=ema(closes,200)
    rs=rsi_fn(closes); mc,sig,hist=macd_fn(closes); bb=bollinger_fn(closes)
    atr_r=[highs[0]-lows[0]]
    for i in range(1,len(closes)):
        tr=max(highs[i]-lows[i],abs(highs[i]-closes[i-1]),abs(lows[i]-closes[i-1]))
        atr_r.append(tr)
    atr_s=sma(atr_r,14)
    out=[]
    for i,row in enumerate(rows):
        out.append({**row,
            "ema20":round(e20[i],2),"ema50":round(e50[i],2),"ema200":round(e200[i],2),
            "rsi":round(rs[i],2) if rs[i] is not None else None,
            "macd":round(mc[i],4),"macd_signal":round(sig[i],4),"macd_hist":round(hist[i],4),
            "bb_upper":bb[i][0],"bb_mid":bb[i][1],"bb_lower":bb[i][2],
            "atr":round(atr_s[i],4) if atr_s[i] is not None else None,
        })
    return out

# ---- LSTM in pure Python ----
_sig = lambda x: 1/(1+math.exp(max(-500,min(500,-x))))
_tanh = lambda x: math.tanh(max(-20,min(20,x)))

def lstm_predict(closes, horizon=7, epochs=25, model_type="lstm"):
    mn,mx=min(closes),max(closes); rng=mx-mn or 1
    norm=[(c-mn)/rng for c in closes]
    seq_len=20; H=20
    random.seed(42)
    rv=lambda s=0.08: (random.random()*2-1)*s
    Wf=[[rv() for _ in range(1+H)] for _ in range(H)]
    Wi=[[rv() for _ in range(1+H)] for _ in range(H)]
    Wc=[[rv() for _ in range(1+H)] for _ in range(H)]
    Wo=[[rv() for _ in range(1+H)] for _ in range(H)]
    bf=[1.0]*H; bi=[0.0]*H; bc=[0.0]*H; bo=[0.0]*H
    Wy=[rv(0.05) for _ in range(H)]; by=0.0
    lr=0.005 if model_type!="ensemble" else 0.003
    X=[norm[i:i+seq_len] for i in range(len(norm)-seq_len-1)]
    y_=[norm[i+seq_len] for i in range(len(norm)-seq_len-1)]
    sample=min(100