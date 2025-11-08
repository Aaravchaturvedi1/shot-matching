// Shot Match Similarity (PURE JS) — robust to missing optional elements
(function () {
  // required elements
  const statusEl  = document.getElementById('status');
  const proVid    = document.getElementById('proVid');
  const userVid   = document.getElementById('userVid');
  const proCanvas = document.getElementById('proCanvas');
  const userCanvas= document.getElementById('userCanvas');
  const analyzeBtn= document.getElementById('analyzeBtn');
  const fileInput = document.getElementById('file'); // your upload input

  // optional elements (guarded)
  const scoreBar  = document.getElementById('scoreBar');

  // small helpers
  const say = (t) => { if (statusEl) statusEl.textContent = t; };

  // ensure libs exist
  if (!(window.tf && window.poseDetection)) {
    say('Libraries not loaded. Check CDN script order.');
    return;
  }

  // upload → set user video
  if (fileInput && userVid) {
    fileInput.addEventListener('change', () => {
      const f = fileInput.files && fileInput.files[0];
      if (!f) return;
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
  
  // 💡 Background gradient for better contrast
  const grad = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
  grad.addColorStop(0, '#0f172a');   // dark navy
  grad.addColorStop(1, '#1e293b');   // slate
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // 💪 Brighter + thicker lines
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

  // 🔵 Joint circles (slightly larger)
  k.forEach(p => {
    if (p.score >= 0.3) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
      ctx.fill();
    }
  });
}


  // Angles
  function angle(a,b,c){
    const ab=[a.x-b.x,a.y-b.y], cb=[c.x-b.x,c.y-b.y];
    const dot=ab[0]*cb[0]+ab[1]*cb[1];
    const na=Math.hypot(...ab), nc=Math.hypot(...cb);
    const cos=Math.max(-1,Math.min(1,dot/(na*nc+1e-6)));
    return Math.acos(cos)*180/Math.PI;
  }
  function kp(obj,name){ return obj.keypoints.find(p=>p.name===name)||obj.keypoints.find(p=>p.part===name); }

  function extractAngles(pose){
    const ls=kp(pose,'left_shoulder'), le=kp(pose,'left_elbow'), lw=kp(pose,'left_wrist');
    const rs=kp(pose,'right_shoulder'), re=kp(pose,'right_elbow'), rw=kp(pose,'right_wrist');
    const lh=kp(pose,'left_hip'), lk=kp(pose,'left_knee'), la=kp(pose,'left_ankle');
    const rh=kp(pose,'right_hip'), rk=kp(pose,'right_knee'), ra=kp(pose,'right_ankle');
    if([ls,le,lw,rs,re,rw,lh,lk,la,rh,rk,ra].some(p=>!p||p.score<0.3)) return null;
    return [
      angle(ls, le, lw),  // left elbow
      angle(rs, re, rw),  // right elbow
      angle(lh, ls, le),  // left shoulder
      angle(rh, rs, re),  // right shoulder
      angle(lh, lk, la),  // left knee
      angle(rh, rk, ra)   // right knee
    ];
  }

  // DTW
  function dtw(a,b,windowRatio=0.2){
    const n=a.length, m=b.length;
    const w=Math.max(Math.floor(Math.max(n,m)*windowRatio), Math.abs(n-m));
    const INF=1e9;
    const D=Array.from({length:n+1},()=>Array(m+1).fill(INF));
    D[0][0]=0;
    const dist=(v1,v2)=> {
      let s=0; for(let i=0;i<v1.length;i++){ const d=v1[i]-v2[i]; s+=d*d; }
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
        const feat = extractAngles(poses[0]);
        if(feat) seq.push(feat);
      }
      say(`Analyzing ${seq.length} frames…`);
    }
    return seq;
  }

  const scoreFromDist = (d) => Math.max(0, Math.min(100, Math.round(100*Math.exp(-d/12))));

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

  // MAIN analyze function (exposed)
  async function analyze(){
    try{
      if(!proVid || !userVid || !proCanvas || !userCanvas){ say('Missing required elements.'); return; }
      if(!userVid.src){ alert('Please choose your video.'); return; }

      say('Loading models…');
      await ensureDetector();

      say('Estimating poses (pro)…');
      const seqPro  = await sampleSequence(proVid,  proCanvas);

      say('Estimating poses (you)…');
      const seqUser = await sampleSequence(userVid, userCanvas);

      if(seqPro.length<8 || seqUser.length<8){
        say('Not enough frames detected. Use a short, clear clip with full body in frame.');
        return;
      }

      say('Computing DTW…');
      const dist = dtw(seqUser, seqPro, 0.15);
      const mainScore = scoreFromDist(dist);

      // UI update
      if (scoreBar) scoreBar.style.width = mainScore + '%';
      say('Done ✅  Score: ' + mainScore);

      // Optional: integrate with Fun Pack if present
      if (window.awardShotResult) {
        // Rough per-area splits (can refine later)
        // Use whole-body score for all three for now
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

  // Attach click handler after DOM is ready
  const attach = () => { if (analyzeBtn) { analyzeBtn.addEventListener('click', analyze); say('Ready. Click ⭐ Analyze Similarity'); } };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', attach);
  } else {
    attach();
  }
})();

