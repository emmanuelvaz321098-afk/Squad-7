"""
ACOUSTIC SENTINEL v3.0
"""
import json,math,time
import numpy as np
from fastapi import FastAPI,WebSocket,WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

SPEED_OF_SOUND=343.0
SAMPLE_RATE=44100
GUNSHOT_RMS_THRESHOLD=0.12
COOLDOWN_FRAMES=30
MAX_EVENTS=200

app=FastAPI(title="Acoustic Sentinel v3")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

events=[]

def rms(s):
    return float(np.sqrt(np.mean(s**2)))

def db(s):
    return 20*math.log10(max(rms(s),1e-10))

def gcc_phat(x,y,max_lag=None):
    n=len(x)+len(y)-1
    n_fft=1<<(n-1).bit_length()
    X,Y=np.fft.rfft(x,n=n_fft),np.fft.rfft(y,n=n_fft)
    G=X*np.conj(Y)
    denom=np.abs(G)
    denom[denom<1e-10]=1e-10
    G/=denom
    cc=np.fft.irfft(G,n=n_fft)
    if max_lag is None:
        max_lag=len(x)//2
    cc=np.concatenate([cc[-max_lag:],cc[:max_lag+1]])
    lag=int(np.argmax(cc))-max_lag
    return lag,float(np.max(cc))

def tdoa_to_doa(lag,spacing):
    tau=lag/SAMPLE_RATE
    c=max(-1.0,min(1.0,(SPEED_OF_SOUND*tau)/spacing))
    return math.degrees(math.acos(c))

def classify(sig_a,sig_b,threshold):
    r=rms(sig_a)
    if r<threshold:
        return False,0.0
    spec=np.abs(np.fft.rfft(sig_a))
    geo=np.exp(np.mean(np.log(spec+1e-10)))
    arith=np.mean(spec)
    conf=0.6*min(r/(threshold*3),1.0)+0.4*min((geo/(arith+1e-10))/0.3,1.0)
    return True,float(conf)

HTML="""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0">
<title>ACOUSTIC SENTINEL v3</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root{--bg:#080b0f;--panel:#0d1117;--border:#1a2535;--accent:#00ff88;--accent2:#ff3c3c;--accent3:#ffb800;--dim:#1e2d3d;--text:#c8d8e8;--text-dim:#4a6070}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:'Share Tech Mono',monospace;min-height:100vh}
header{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid var(--border);background:rgba(13,17,23,.97);position:sticky;top:0;z-index:1000;flex-wrap:wrap;gap:8px}
.logo{display:flex;align-items:center;gap:12px}
.logo-icon{width:32px;height:32px;border:2px solid var(--accent);border-radius:50%;display:flex;align-items:center;justify-content:center;position:relative;box-shadow:0 0 20px rgba(0,255,136,.3);flex-shrink:0}
.logo-icon::before{content:'';width:7px;height:7px;background:var(--accent);border-radius:50%;animation:pulse 2s infinite}
.logo-icon::after{content:'';position:absolute;width:44px;height:44px;border:1px solid rgba(0,255,136,.2);border-radius:50%;animation:radar-ring 2s infinite}
@keyframes radar-ring{0%{transform:scale(.7);opacity:.8}100%{transform:scale(1.4);opacity:0}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
@keyframes flash{from{opacity:1}to{opacity:.4}}
@keyframes alarmPulse{0%,100%{background:rgba(255,60,60,0)}50%{background:rgba(255,60,60,0.15)}}
.logo-text{font-family:'Orbitron',monospace;font-weight:900;font-size:14px;letter-spacing:3px;color:var(--accent)}
.logo-sub{font-size:8px;color:var(--text-dim);letter-spacing:2px}
.hdr-right{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.status-item{text-align:center;font-size:9px;color:var(--text-dim)}
.status-val{display:block;font-family:'Orbitron',monospace;font-size:12px;color:var(--accent)}
#system-status{display:flex;align-items:center;gap:6px;font-size:10px;letter-spacing:2px}
.dot{width:7px;height:7px;border-radius:50%;background:var(--text-dim);transition:all .3s;flex-shrink:0}
.dot.active{background:var(--accent);box-shadow:0 0 20px rgba(0,255,136,.3);animation:pulse 1.5s infinite}
.dot.alert{background:var(--accent2);box-shadow:0 0 20px rgba(255,60,60,.5);animation:pulse .5s infinite}
.dot.warn{background:var(--accent3);animation:pulse 1s infinite}
.tabs{display:flex;border-bottom:1px solid var(--border);background:var(--panel);position:sticky;top:65px;z-index:999}
.tab{flex:1;padding:12px 4px;text-align:center;font-family:'Orbitron',monospace;font-size:9px;letter-spacing:2px;color:var(--text-dim);cursor:pointer;border-bottom:2px solid transparent;transition:all .2s;-webkit-tap-highlight-color:transparent}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab-content{display:none}
.tab-content.active{display:block}
#alarm-overlay{display:none;position:fixed;inset:0;z-index:9999;pointer-events:none;animation:alarmPulse .4s infinite}
#alarm-overlay.active{display:block}
.panel{border:1px solid var(--border);background:var(--panel);position:relative;overflow:hidden}
.panel::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--accent),transparent);opacity:.4}
.panel-label{position:absolute;top:10px;left:14px;font-size:9px;letter-spacing:3px;color:var(--text-dim);z-index:2}
.panel-label span{color:var(--accent)}
.doa-panel{display:flex;flex-direction:column;align-items:center;padding:40px 16px 20px;gap:16px}
.polar-wrap{position:relative;width:min(320px,88vw);height:min(320px,88vw)}
#polar{width:100%;height:100%}
.doa-readout{text-align:center;width:100%}
.doa-angle{font-family:'Orbitron',monospace;font-size:44px;font-weight:900;color:var(--accent);line-height:1}
.doa-label{font-size:9px;color:var(--text-dim);letter-spacing:3px;margin-top:4px}
.tdoa-display{display:flex;gap:20px;margin-top:10px;justify-content:center}
.tdoa-item{text-align:center}
.tdoa-val{font-family:'Orbitron',monospace;font-size:13px;color:var(--accent3)}
.tdoa-lbl{font-size:8px;color:var(--text-dim)}
.detect-panel{padding:40px 16px 16px}
.detect-status{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
.detect-badge{font-family:'Orbitron',monospace;font-size:10px;letter-spacing:2px;padding:5px 12px;border:1px solid var(--text-dim);color:var(--text-dim);transition:all .3s}
.detect-badge.gunshot{border-color:var(--accent2);color:var(--accent2);box-shadow:0 0 20px rgba(255,60,60,.5);animation:flash .3s ease infinite alternate}
.detect-badge.clear{border-color:var(--accent);color:var(--accent)}
.confidence-bar{height:4px;background:var(--dim);border-radius:2px;overflow:hidden;margin-bottom:10px}
.confidence-fill{height:100%;background:linear-gradient(90deg,var(--accent),var(--accent3));width:0%;transition:width .4s ease;border-radius:2px}
.conf-label{font-size:9px;color:var(--text-dim);letter-spacing:2px}
.conf-val{float:right;color:var(--accent3)}
#waveform{width:100%;height:65px;display:block;border:1px solid var(--dim);border-radius:2px;margin-top:10px}
.mic-panel{padding:40px 16px 16px}
.mic-array{display:flex;gap:10px;align-items:center;margin-bottom:14px}
.mic-unit{flex:1;border:1px solid var(--dim);padding:10px;text-align:center;transition:all .3s}
.mic-unit.active{border-color:var(--accent);box-shadow:0 0 20px rgba(0,255,136,.3)}
.mic-name{font-size:9px;color:var(--text-dim);letter-spacing:2px}
.mic-level{font-family:'Orbitron',monospace;font-size:15px;color:var(--accent);margin:4px 0}
.mic-bar{height:3px;background:var(--dim);border-radius:2px;overflow:hidden}
.mic-bar-fill{height:100%;background:var(--accent);width:0%;transition:width .1s;border-radius:2px}
.mic-arrow{font-size:18px;color:var(--text-dim);flex-shrink:0}
.spacing-control{display:flex;align-items:center;gap:10px;font-size:10px;color:var(--text-dim);margin-bottom:12px}
.spacing-control input[type=range]{flex:1;-webkit-appearance:none;height:2px;background:var(--dim);outline:none;border-radius:2px}
.spacing-control input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;background:var(--accent);border-radius:50%;cursor:pointer}
.spacing-val{color:var(--accent);min-width:50px}
.btn{width:100%;padding:12px;font-family:'Orbitron',monospace;font-size:10px;letter-spacing:3px;border:1px solid var(--accent);background:transparent;color:var(--accent);cursor:pointer;transition:all .2s;touch-action:manipulation;margin-bottom:6px}
.btn.active{background:rgba(0,255,136,.1)}
.btn.danger{border-color:var(--accent2);color:var(--accent2)}
.btn.warn{border-color:var(--accent3);color:var(--accent3)}
.btn.on{background:rgba(255,184,0,.15);border-color:var(--accent3);color:var(--accent3)}
.sim-row{display:flex;gap:6px;margin-top:4px}
.btn-sm{padding:10px 4px;font-size:8px;flex:1;margin-bottom:0}
#map{height:350px;width:100%;z-index:1}
.map-panel{padding:40px 0 0}
.map-controls{padding:12px 16px;display:flex;gap:8px;flex-wrap:wrap}
.map-btn{flex:1;padding:8px;font-family:'Orbitron',monospace;font-size:8px;letter-spacing:2px;border:1px solid var(--border);background:var(--panel);color:var(--text-dim);cursor:pointer;min-width:80px;touch-action:manipulation}
.shot-count-badge{display:inline-block;background:var(--accent2);color:#fff;font-family:'Orbitron',monospace;font-size:9px;padding:2px 8px;border-radius:2px;margin-left:8px}
.tracker-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px}
.tracker-card{border:1px solid var(--border);padding:12px;text-align:center;transition:all .3s}
.tracker-card.active{border-color:var(--accent2);box-shadow:0 0 15px rgba(255,60,60,.2)}
.tracker-card.inactive{opacity:.4}
.tc-id{font-family:'Orbitron',monospace;font-size:18px;font-weight:900;color:var(--accent2)}
.tc-angle{font-family:'Orbitron',monospace;font-size:24px;font-weight:900;color:var(--accent)}
.tc-time{font-size:8px;color:var(--text-dim);margin-top:4px}
.tc-conf{font-size:9px;color:var(--accent3)}
.tracker-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:12px}
.stat-card{border:1px solid var(--border);padding:10px;text-align:center}
.stat-val{font-family:'Orbitron',monospace;font-size:20px;color:var(--accent)}
.stat-lbl{font-size:8px;color:var(--text-dim);letter-spacing:1px;margin-top:2px}
.stat-card.danger .stat-val{color:var(--accent2)}
.stat-card.warn .stat-val{color:var(--accent3)}
.log-panel{padding:40px 0 0}
.log-header{padding:0 16px 12px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--dim)}
.log-count{font-family:'Orbitron',monospace;font-size:20px;color:var(--accent2)}
.log-count-lbl{font-size:9px;color:var(--text-dim);letter-spacing:2px;display:block}
.log-list{max-height:300px;overflow-y:auto}
.log-entry{padding:10px 16px;border-bottom:1px solid rgba(26,37,53,.5);display:grid;grid-template-columns:1fr auto;gap:4px;animation:slideIn .3s ease}
@keyframes slideIn{from{opacity:0;transform:translateX(8px)}to{opacity:1;transform:translateX(0)}}
.log-entry.gunshot{border-left:3px solid var(--accent2)}
.log-type{font-family:'Orbitron',monospace;font-size:10px;letter-spacing:2px;color:var(--accent2)}
.log-details{font-size:9px;color:var(--text-dim);margin-top:2px}
.log-time{font-size:9px;color:var(--text-dim);text-align:right}
.log-angle{font-family:'Orbitron',monospace;font-size:11px;color:var(--accent3)}
.ws-bar{position:fixed;bottom:0;left:0;right:0;background:var(--panel);border-top:1px solid var(--border);padding:8px 16px;display:flex;align-items:center;gap:8px;font-size:9px;letter-spacing:2px;color:var(--text-dim);z-index:998}
.leaflet-container{background:#0d1117!important}
.leaflet-tile{filter:invert(1) hue-rotate(180deg) brightness(0.7) contrast(1.2)}
</style>
</head>
<body>
<div id="alarm-overlay"></div>
<header>
<div class="logo">
<div class="logo-icon"></div>
<div>
<div class="logo-text">ACOUSTIC SENTINEL <span style="color:var(--accent3);font-size:10px">v3</span></div>
<div class="logo-sub">DOA - MAP - MULTI-TRACK - ALARM</div>
</div>
</div>
<div class="hdr-right">
<div class="status-item"><span class="status-val" id="hdr-time">--:--:--</span>UTC</div>
<div class="status-item"><span class="status-val" id="hdr-events">0</span>SHOTS</div>
<div id="system-status"><div class="dot" id="sys-dot"></div><span id="sys-label">STANDBY</span></div>
</div>
</header>
<div class="tabs">
<div class="tab active" onclick="switchTab('doa')">DOA</div>
<div class="tab" onclick="switchTab('map')">MAP</div>
<div class="tab" onclick="switchTab('tracker')">TRACKER</div>
<div class="tab" onclick="switchTab('log')">LOG</div>
</div>
<div id="tab-doa" class="tab-content active">
<div class="panel doa-panel">
<div class="panel-label"><span>01</span> / POLAR DOA</div>
<div class="polar-wrap"><canvas id="polar" width="320" height="320"></canvas></div>
<div class="doa-readout">
<div class="doa-angle" id="doa-angle">---</div>
<div class="doa-label">DIRECTION OF ARRIVAL</div>
<div class="tdoa-display">
<div class="tdoa-item"><div class="tdoa-val" id="tdoa-val">0.000</div><div class="tdoa-lbl">TDOA ms</div></div>
<div class="tdoa-item"><div class="tdoa-val" id="snr-val">--</div><div class="tdoa-lbl">SNR dB</div></div>
<div class="tdoa-item"><div class="tdoa-val" id="dist-val">0.50</div><div class="tdoa-lbl">MIC m</div></div>
</div>
</div>
</div>
<div class="panel detect-panel">
<div class="panel-label"><span>02</span> / CLASSIFICATION</div>
<div class="detect-status">
<div class="detect-badge" id="detect-badge">MONITORING</div>
<div style="text-align:right;font-size:9px;color:var(--text-dim)">
<div>LAST: <span id="last-detect" style="color:var(--text)">--</span></div>
<div>PEAK: <span id="peak-db" style="color:var(--accent3)">-- dB</span></div>
</div>
</div>
<div class="conf-label">CONFIDENCE <span class="conf-val" id="conf-pct">0%</span></div>
<div class="confidence-bar"><div class="confidence-fill" id="conf-fill"></div></div>
<canvas id="waveform" width="340" height="65"></canvas>
</div>
<div class="panel mic-panel">
<div class="panel-label"><span>03</span> / ARRAY AND CONTROLS</div>
<div class="mic-array">
<div class="mic-unit" id="mic0"><div class="mic-name">MIC A</div><div class="mic-level" id="mic0-lvl">-inf</div><div class="mic-bar"><div class="mic-bar-fill" id="mic0-bar"></div></div></div>
<div class="mic-arrow">---</div>
<div class="mic-unit" id="mic1"><div class="mic-name">MIC B</div><div class="mic-level" id="mic1-lvl">-inf</div><div class="mic-bar"><div class="mic-bar-fill" id="mic1-bar"></div></div></div>
</div>
<div class="spacing-control">SPACING <input type="range" id="spacing-slider" min="0.1" max="2.0" step="0.05" value="0.5"> <span class="spacing-val" id="spacing-val">0.50 m</span></div>
<button class="btn" id="start-btn" onclick="toggleListening()">START LISTENING</button>
<button class="btn warn" id="alarm-btn" onclick="toggleAlarm()">ALARM: OFF</button>
<div class="sim-row">
<button class="btn-sm btn" onclick="simGunshot(45)">SIM 45</button>
<button class="btn-sm btn" onclick="simGunshot(90)">SIM 90</button>
<button class="btn-sm btn" onclick="simGunshot(135)">SIM 135</button>
<button class="btn-sm btn danger" onclick="clearAll()">CLEAR</button>
</div>
</div>
</div>
<div id="tab-map" class="tab-content">
<div class="panel map-panel">
<div class="panel-label"><span>02</span> / GUNSHOT MAP <span id="map-shot-count" class="shot-count-badge">0 SHOTS</span></div>
<div id="map"></div>
<div class="map-controls">
<button class="map-btn" onclick="centerMap()">MY LOCATION</button>
<button class="map-btn" onclick="clearMapMarkers()">CLEAR PINS</button>
<button class="map-btn" onclick="simGunshot(Math.random()*180)">SIM SHOT</button>
</div>
<div style="padding:0 16px 12px;font-size:9px;color:var(--text-dim)">Map pins show estimated direction from your location.</div>
</div>
</div>
<div id="tab-tracker" class="tab-content">
<div style="padding:16px 16px 8px">
<div class="tracker-stats">
<div class="stat-card danger"><div class="stat-val" id="stat-total">0</div><div class="stat-lbl">TOTAL SHOTS</div></div>
<div class="stat-card warn"><div class="stat-val" id="stat-last-angle">--</div><div class="stat-lbl">LAST ANGLE</div></div>
<div class="stat-card"><div class="stat-val" id="stat-avg-conf">--</div><div class="stat-lbl">AVG CONF</div></div>
</div>
<div style="font-family:'Orbitron',monospace;font-size:9px;letter-spacing:3px;color:var(--text-dim);margin-bottom:10px">RECENT DETECTIONS</div>
<div class="tracker-grid">
<div class="tracker-card inactive" id="tc-0"><div class="tc-id">#--</div><div class="tc-angle">---</div><div class="tc-conf">--</div><div class="tc-time">--</div></div>
<div class="tracker-card inactive" id="tc-1"><div class="tc-id">#--</div><div class="tc-angle">---</div><div class="tc-conf">--</div><div class="tc-time">--</div></div>
<div class="tracker-card inactive" id="tc-2"><div class="tc-id">#--</div><div class="tc-angle">---</div><div class="tc-conf">--</div><div class="tc-time">--</div></div>
<div class="tracker-card inactive" id="tc-3"><div class="tc-id">#--</div><div class="tc-angle">---</div><div class="tc-conf">--</div><div class="tc-time">--</div></div>
</div>
<div style="font-family:'Orbitron',monospace;font-size:9px;letter-spacing:3px;color:var(--text-dim);margin-bottom:10px">DIRECTION HISTOGRAM</div>
<canvas id="histogram" width="340" height="120" style="width:100%;border:1px solid var(--dim);border-radius:2px"></canvas>
</div>
</div>
<div id="tab-log" class="tab-content">
<div class="panel log-panel">
<div class="panel-label"><span>04</span> / EVENT LOG</div>
<div class="log-header">
<div><span class="log-count-lbl">TOTAL</span><span class="log-count" id="log-count">0</span></div>
<button onclick="exportCSV()" style="font-family:'Orbitron',monospace;font-size:8px;letter-spacing:2px;padding:4px 10px;border:1px solid var(--accent);background:transparent;color:var(--accent);cursor:pointer">EXPORT CSV</button>
</div>
<div class="log-list" id="log-list"></div>
</div>
</div>
<div class="ws-bar">
<div class="dot" id="ws-dot"></div>
<span id="ws-label">CONNECTING...</span>
</div>
<script>
var SPEED=343,API=window.location.origin;
var WS_URL=(location.protocol==='https:'?'wss:':'ws:')+'//'+location.host+'/ws/audio';
var ws,audioCtx,analyserA,analyserB,stream;
var isListening=false,animFrame,micSpacing=0.5;
var doaAngle=null,tdoaMs=0,detecting=false,detectTimeout;
var eventCount=0,lastPing=Date.now();
var doaHistory=[],HIST=8,particles=[];
var alarmEnabled=false,alarmCtx=null;
var userLat=null,userLng=null;
var mapObj=null,mapMarkers=[],userMarker=null;
var allEvents=[];
var histogramData=new Array(18).fill(0);

function switchTab(name){
var names=['doa','map','tracker','log'];
document.querySelectorAll('.tab').forEach(function(t,i){t.classList.toggle('active',names[i]===name)});
document.querySelectorAll('.tab-content').forEach(function(c){c.classList.remove('active')});
document.getElementById('tab-'+name).classList.add('active');
if(name==='map')setTimeout(function(){if(mapObj)mapObj.invalidateSize()},100);
if(name==='tracker')drawHistogram();
}

function initMap(){
mapObj=L.map('map',{zoomControl:true}).setView([0,0],2);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'OpenStreetMap',maxZoom:19}).addTo(mapObj);
if(navigator.geolocation){
navigator.geolocation.getCurrentPosition(function(pos){
userLat=pos.coords.latitude;userLng=pos.coords.longitude;
mapObj.setView([userLat,userLng],15);
userMarker=L.circleMarker([userLat,userLng],{radius:10,color:'#00ff88',fillColor:'#00ff88',fillOpacity:0.8,weight:2}).addTo(mapObj).bindPopup('YOUR LOCATION');
},function(){});
}
}
initMap();
function centerMap(){if(userLat&&userLng)mapObj.setView([userLat,userLng],15)}

function addMapMarker(angle,confidence,eventId){
if(!userLat||!userLng)return;
var dist=50+(1-confidence)*200;
var rad=(angle-90)*Math.PI/180;
var latOff=dist/111000*Math.cos(rad);
var lngOff=dist/(111000*Math.cos(userLat*Math.PI/180))*Math.sin(rad);
var lat=userLat+latOff,lng=userLng+lngOff;
var marker=L.circleMarker([lat,lng],{radius:8+confidence*6,color:'#ff3c3c',fillColor:'#ff3c3c',fillOpacity:0.6+confidence*0.3,weight:
