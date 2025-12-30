// Shot Match Similarity (MULTI-METRIC WEIGHTED ANALYSIS)
(function () {
  // required elements
  const statusEl  = document.getElementById('status');
  const proVid    = document.getElementById('proVid');
  const userVid   = document.getElementById('userVid');
  const proCanvas = document.getElementById('proCanvas');
  const userCanvas= document.getElementById('userCanvas');
  const analyzeBtn= document.getElementById('analyzeBtn');
  const fileInput = document.getElementById('file');

  // optional elements (guarded)
  const scoreBar  = document.getElementById('scoreBar');

  // small helpers
  const say = (t) => { if (statusEl) statusEl.textContent = t; };

  // ensure libs exist
  if (!(window.tf && window.poseDetection)) {
    say('Libraries not loaded. Check CDN script order.');
    return;
  }

  // 🎯 WEIGHTS for different metrics (must sum to 1.0)
  const WEIGHTS = {
    angles: 0.20,              // Joint angles (less important now)
    relativePositions: 0.30,   // Most important - body geometry
    velocity: 0.25,            // Speed/smoothness of movement
    SequentialTiming: 0.15,    // Kinetic chain timing
    bodyProportions: 0.10      // Ratios and balance
  };

  // --- SAME VIDEO DETECTION HELPERS ---
  function proFileNameFromSrc() {
    try {
      const u = new URL(proVid.currentSrc || proVid.src, location.href);
      const last = u.pathname.split('/').pop() || '';
      return last.split('?')[0].toLowerCase();
    } catch { return ''; }
  }

  async function proContentLength() {
    try {
      const r = await fetch(proVid.currentSrc || proVid.src, { method: 'HEAD' });
      const len = r.headers.get('content-length');
      return len ? parseInt(len, 10) : null;
    } catch { return null; }
  }

  async function isSameVideoByMeta(userFile) {
    if (!userFile) return false;
    const proLen = await proContentLength();
    const nameMatch = proFileNameFromSrc() === (userFile.name || '').toLowerCase();
    const sizeMatch = (proLen != null) && Math.abs(proLen - userFile.size) <= 16;
    return nameMatch || sizeMatch;
  }

  // upload → set user video
  if (fileInput && userVid) {
    fileInput.addEventListener('change', () => {
      const f = fileInput.files && fileInput.files[0];
      if (!f) return;
      userVid._file = f;
      if (userVid._u) URL.revokeObjectURL(userVid._u);
      userVid._u = URL.createObjectURL(f);
      userVid.src = userVid._u;
      userVid.load();
    });
  }

  // Drawing utilities
  function drawPose(canvas, pose, color = '#00ff99') {
    const ctx = canvas.getContext('2d');
    const k = pose.keypoints || pose.keypoints3D || [];
    
    const grad = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    grad.addColorStop(0, '#0f172a');
    grad.addColorStop(1, '#1e293b');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 5;
    ctx.strokeStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 10;
    ctx.fillStyle = color;

    const pairs = [
      ['left_shoulder','right_shoulder'], ['left_hip','right_hip'],
      ['left_shoulder','left_elbow'], ['left_elbow','left_wrist'],
      ['right_shoulder','right_elbow'], ['right_elbow','right_wrist'],
      ['left_hip','left_knee'], ['left_knee','left_ankle'],
      ['right_hip','right_knee'], ['right_knee','right_ankle']
    ];
    const gp = (name) => k.find(p => p.name === name) || k.find(p => p.part === name);

    pairs.forEach(([a,b]) => {
      const A = gp(a), B = gp(b);
      if (!A || !B || A.score < 0.3 || B.score < 0.3) return;
      ctx.beginPath();
      ctx.moveTo(A.x, A.y);
      ctx.lineTo(B.x, B.y);
      ctx.stroke();
    });

    k.forEach(p => {
      if (p.score >= 0.3) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  }

  // === HELPER FUNCTIONS ===
  function kp(obj,name){ 
    return obj.keypoints.find(p=>p.name===name)||obj.keypoints.find(p=>p.part===name); 
  }

  function angle(a,b,c){
    const ab=[a.x-b.x,a.y-b.y], cb=[c.x-b.x,c.y-b.y];
    const dot=ab[0]*cb[0]+ab[1]*cb[1];
    const na=Math.hypot(...ab), nc=Math.hypot(...cb);
    const cos=Math.max(-1,Math.min(1,dot/(na*nc+1e-6)));
    return Math.acos(cos)*180/Math.PI;
  }

  function dist2d(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  // === FEATURE EXTRACTION ===

  // 1️⃣ Joint Angles (original metric)
  function extractAngles(pose){
    const ls=kp(pose,'left_shoulder'), le=kp(pose,'left_elbow'), lw=kp(pose,'left_wrist');
    const rs=kp(pose,'right_shoulder'), re=kp(pose,'right_elbow'), rw=kp(pose,'right_wrist');
    const lh=kp(pose,'left_hip'), lk=kp(pose,'left_knee'), la=kp(pose,'left_ankle');
    const rh=kp(pose,'right_hip'), rk=kp(pose,'right_knee'), ra=kp(pose,'right_ankle');
    if([ls,le,lw,rs,re,rw,lh,lk,la,rh,rk,ra].some(p=>!p||p.score<0.3)) return null;
    return [
      angle(ls, le, lw),  angle(rs, re, rw),
      angle(lh, ls, le),  angle(rh, rs, re),
      angle(lh, lk, la),  angle(rh, rk, ra)
    ];
  }

  // 2️⃣ Relative Joint Positions (normalized by body scale)
  function extractRelativePositions(pose){
    const ls=kp(pose,'left_shoulder'), rs=kp(pose,'right_shoulder');
    const lh=kp(pose,'left_hip'), rh=kp(pose,'right_hip');
    const lw=kp(pose,'left_wrist'), rw=kp(pose,'right_wrist');
    const lk=kp(pose,'left_knee'), rk=kp(pose,'right_knee');
    
    if([ls,rs,lh,rh,lw,rw,lk,rk].some(p=>!p||p.score<0.3)) return null;
    
    // Body center and scale
    const cx = (ls.x + rs.x + lh.x + rh.x) / 4;
    const cy = (ls.y + rs.y + lh.y + rh.y) / 4;
    const scale = dist2d(ls, rs) + dist2d(lh, rh) + 1e-6;
    
    // Normalized positions relative to body center
    const normalize = (p) => [(p.x - cx)/scale, (p.y - cy)/scale];
    
    return [
      ...normalize(lw), ...normalize(rw),  // wrist positions
      ...normalize(lk), ...normalize(rk),  // knee positions
      ...normalize(ls), ...normalize(rs),  // shoulder positions
      ...normalize(lh), ...normalize(rh)   // hip positions
    ];
  }

  // 3️⃣ Body Proportions & Ratios
  function extractBodyProportions(pose){
    const ls=kp(pose,'left_shoulder'), rs=kp(pose,'right_shoulder');
    const lh=kp(pose,'left_hip'), rh=kp(pose,'right_hip');
    const lw=kp(pose,'left_wrist'), rw=kp(pose,'right_wrist');
    
    if([ls,rs,lh,rh,lw,rw].some(p=>!p||p.score<0.3)) return null;
    
    const shoulderWidth = dist2d(ls, rs);
    const hipWidth = dist2d(lh, rh);
    const torsoHeight = (dist2d(ls, lh) + dist2d(rs, rh)) / 2;
    const leftArmReach = dist2d(ls, lw);
    const rightArmReach = dist2d(rs, rw);
    
    return [
      hipWidth / (shoulderWidth + 1e-6),      // hip-shoulder ratio
      leftArmReach / (torsoHeight + 1e-6),    // left arm extension
      rightArmReach / (torsoHeight + 1e-6),   // right arm extension
      (lw.y - ls.y) / (torsoHeight + 1e-6),   // left wrist height
      (rw.y - rs.y) / (torsoHeight + 1e-6)    // right wrist height
    ];
  }

  // === COMPREHENSIVE FEATURE EXTRACTION ===
  function extractAllFeatures(pose){
    const angles = extractAngles(pose);
    const positions = extractRelativePositions(pose);
    const proportions = extractBodyProportions(pose);
    
    if(!angles || !positions || !proportions) return null;
    
    return {
      angles,
      positions,
      proportions
    };
  }

  // === VELOCITY & TIMING CALCULATIONS ===
  
  // Calculate velocity between consecutive frames
  function calculateVelocities(sequence) {
    const velocities = [];
    for(let i = 1; i < sequence.length; i++) {
      const prev = sequence[i-1];
      const curr = sequence[i];
      
      // Velocity = change in position features
      const vel = curr.positions.map((v, idx) => v - prev.positions[idx]);
      velocities.push(vel);
    }
    return velocities;
  }

  // Calculate timing of key events (peaks, transitions)
  function calculateSequentialTiming(sequence) {
    const timings = [];
    
    for(let i = 0; i < sequence.length; i++) {
      const feat = sequence[i];
      
      // Key timing indicators
      timings.push([
        feat.proportions[1],  // left arm extension (backswing/follow-through)
        feat.proportions[2],  // right arm extension
        feat.positions[0],    // left wrist X (swing path)
        feat.positions[4]     // left shoulder X (rotation)
      ]);
    }
    return timings;
  }

  // === DTW FOR DIFFERENT FEATURE TYPES ===
  
  function dtwSimple(a, b, windowRatio=0.4){
    const n=a.length, m=b.length;
    const w=Math.max(Math.floor(Math.max(n,m)*windowRatio), Math.abs(n-m));
    const INF=1e9;
    const D=Array.from({length:n+1},()=>Array(m+1).fill(INF));
    D[0][0]=0;
    
    const dist=(v1,v2)=> {
      let s=0; 
      for(let i=0;i<v1.length;i++){ 
        const d=v1[i]-v2[i]; 
        s+=d*d; 
      }
      return Math.sqrt(s);
    };
    
    for(let i=1;i<=n;i++){
      const jStart=Math.max(1,i-w), jEnd=Math.min(m,i+w);
      for(let j=jStart;j<=jEnd;j++){
        const cost=dist(a[i-1], b[j-1]);
        D[i][j]=cost+Math.min(D[i-1][j], D[i][j-1], D[i-1][j-1]);
      }
    }
    return D[n][m]/(n+m);
  }

  // === MULTI-METRIC COMPARISON ===
  
  function compareSequences(seqUser, seqPro) {
    const results = {};
    
    // 1. Angle comparison
    const anglesUser = seqUser.map(f => f.angles);
    const anglesPro = seqPro.map(f => f.angles);
    results.anglesDist = dtwSimple(anglesUser, anglesPro, 0.4);
    
    // 2. Relative positions comparison
    const posUser = seqUser.map(f => f.positions);
    const posPro = seqPro.map(f => f.positions);
    results.positionsDist = dtwSimple(posUser, posPro, 0.4);
    
    // 3. Body proportions comparison
    const propUser = seqUser.map(f => f.proportions);
    const propPro = seqPro.map(f => f.proportions);
    results.proportionsDist = dtwSimple(propUser, propPro, 0.4);
    
    // 4. Velocity comparison
    const velUser = calculateVelocities(seqUser);
    const velPro = calculateVelocities(seqPro);
    results.velocityDist = dtwSimple(velUser, velPro, 0.4);
    
    // 5. Sequential timing comparison
    const timingUser = calculateSequentialTiming(seqUser);
    const timingPro = calculateSequentialTiming(seqPro);
    results.timingDist = dtwSimple(timingUser, timingPro, 0.4);
    
    return results;
  }

  // === WEIGHTED SCORING ===
  
  function calculateWeightedScore(distances) {
    console.log(distances)
    // Convert each distance to a similarity score (0-100)
    // Using gentler curves for each metric
    // Add safety checks for undefined/null values
    const angleScore = 100 * Math.exp(-(distances.anglesDist || 0) / 25);
    const posScore = 100 * Math.exp(-(distances.positionsDist || 0) / 2.5);
    const propScore = 100 * Math.exp(-(distances.proportionsDist || 0) / 3.0);
    const velScore = 100 * Math.exp(-(distances.velocityDist || 0) / 1.5);
    const timingScore = 100 * Math.exp(-(distances.timingDist || 0) / 2.0);
    
    // Weighted average
    const finalScore = 
      angleScore * WEIGHTS.angles +
      posScore * WEIGHTS.relativePositions +
      velScore * WEIGHTS.velocity +
      timingScore * WEIGHTS.SequentialTiming +  // Fixed typo: was sequentialTiming, should match WEIGHTS key
      propScore * WEIGHTS.bodyProportions;
    
    // Safety check for final score
 
    const safeFinal = isNaN(finalScore) ? 0 : finalScore;
    console.log(safeFinal)

    return {
      final: Math.round(safeFinal),
      breakdown: {
        angles: Math.round(angleScore),
        positions: Math.round(posScore),
        proportions: Math.round(propScore),
        velocity: Math.round(velScore),
        timing: Math.round(timingScore)
      }
    };
  }

  // === SAMPLING ===
  
  async function sampleSequence(video, canvas, take=90){
    await video.play(); video.pause();
    const w=canvas.width = video.videoWidth || 854;
    const h=canvas.height= video.videoHeight|| 480;
    const ctx=canvas.getContext('2d');
    const total = Math.max(1, Math.floor(video.duration*30));
    const step = Math.max(1, Math.floor(total/take));
    const seq=[];
    
    for(let f=0; f<total; f+=step){
      video.currentTime = f/30;
      await new Promise(r=> video.onseeked = ()=>r());
      ctx.drawImage(video,0,0,w,h);
      const poses = await detector.estimatePoses(canvas,{flipHorizontal:false});
      if(poses && poses[0]){
        drawPose(canvas, poses[0]);
        const feat = extractAllFeatures(poses[0]);
        if(feat) seq.push(feat);
      }
      say(`Analyzing ${seq.length} frames…`);
    }
    return seq;
  }

  // detector (create once)
  let detector;

  async function ensureDetector(){
    if(detector) return detector;
    await tf.setBackend('webgl');
    detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.BlazePose,
      { runtime:'mediapipe', modelType:'full', solutionPath:'https://cdn.jsdelivr.net/npm/@mediapipe/pose' }
    );
    return detector;
  }

  // === MAIN ANALYSIS ===
  
  async function analyze(){
    try{
      if(!proVid || !userVid || !proCanvas || !userCanvas){ 
        say('Missing required elements.'); 
        return; 
      }
      if(!userVid.src){ 
        alert('Please choose your video.'); 
        return; 
      }

      say('Loading models…');
      await ensureDetector();

      say('Analyzing pro video…');
      const seqPro  = await sampleSequence(proVid,  proCanvas);

      say('Analyzing your video…');
      const seqUser = await sampleSequence(userVid, userCanvas);

      if(seqPro.length<8 || seqUser.length<8){
        say('Not enough frames detected. Use a short, clear clip with full body in frame.');
        return;
      }

      say('Computing multi-metric comparison…');
      
      // Check for exact match first
      const metaSame = await isSameVideoByMeta(userVid._file);
      if (metaSame) {
        if (scoreBar) scoreBar.style.width = '100%';
        say('Perfect match! ✅ Score: 100');
        if (window.awardShotResult) {
          window.awardShotResult({ timingScore:1, stanceScore:1, swingScore:1 });
        }
        return;
      }
      
      // Multi-metric comparison
      const distances = compareSequences(seqUser, seqPro);
      const scoreResult = calculateWeightedScore(distances);
      
      const mainScore = scoreResult.final;
      const breakdown = scoreResult.breakdown;

      // UI update
      if (scoreBar) scoreBar.style.width = mainScore + '%';
      say(`Done ✅ Score: ${mainScore} | Angles:${breakdown.angles} Pos:${breakdown.positions} Vel:${breakdown.velocity} Time:${breakdown.timing} Props:${breakdown.proportions}`);

      if (window.awardShotResult) {
        const t = mainScore/100, s = mainScore/100, sw = mainScore/100;
        window.awardShotResult({ timingScore:t, stanceScore:s, swingScore:sw });
      }

    }catch(e){
      console.error(e);
      say('Error: ' + (e && e.message ? e.message : e));
    }
  }

  // Expose for console/testing
  window.__analyzeSimilarity = analyze;

  // Attach click handler
  const attach = () => { 
    if (analyzeBtn) { 
      analyzeBtn.addEventListener('click', analyze); 
      say('Ready. Click ⭐ Analyze Similarity'); 
    } 
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', attach);
  } else {
    attach();
  }
})();
