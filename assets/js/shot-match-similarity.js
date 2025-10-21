<!-- TFJS + Pose Detection (BlazePose via MediaPipe runtime) -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection"></script>

<script>
(async function(){
  const statusEl = document.getElementById('status');
  const proVid = document.getElementById('proVid');
  const userVid = document.getElementById('userVid');
  const proCanvas = document.getElementById('proCanvas');
  const userCanvas = document.getElementById('userCanvas');
  const proUrl = document.getElementById('proUrl');
  const userFile = document.getElementById('userFile');
  const analyzeBtn = document.getElementById('analyzeBtn');

  const scoreNum = document.getElementById('scoreNum');
  const scoreBar = document.getElementById('scoreBar');
  const timingSub = document.getElementById('timingSub');
  const stanceSub = document.getElementById('stanceSub');
  const swingSub = document.getElementById('swingSub');

  // Load backend
  await tf.setBackend('webgl');

  // Create detector (BlazePose + MediaPipe runtime: fast & reliable in browser)
  const detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.BlazePose,
    { runtime:'mediapipe', modelType:'full', solutionPath:'https://cdn.jsdelivr.net/npm/@mediapipe/pose' }
  );

  // Video loading helpers
  function fileToURL(file){ return URL.createObjectURL(file); }
  userFile.addEventListener('change', () => {
    if(userFile.files[0]) userVid.src = fileToURL(userFile.files[0]);
  });

  // Drawing utilities
  function drawPose(canvas, pose, color='#5ee1b8'){
    const ctx = canvas.getContext('2d');
    const k = pose.keypoints || pose.keypoints3D || [];
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.lineWidth = 3; ctx.strokeStyle = color; ctx.fillStyle = color;
    const pairs = [
      ['left_shoulder','right_shoulder'], ['left_hip','right_hip'],
      ['left_shoulder','left_elbow'], ['left_elbow','left_wrist'],
      ['right_shoulder','right_elbow'], ['right_elbow','right_wrist'],
      ['left_hip','left_knee'], ['left_knee','left_ankle'],
      ['right_hip','right_knee'], ['right_knee','right_ankle']
    ];
    function gp(name){ return k.find(p=>p.name===name) || k.find(p=>p.part===name); }
    pairs.forEach(([a,b])=>{
      const A = gp(a), B = gp(b); if(!A||!B||A.score<0.3||B.score<0.3) return;
      ctx.beginPath(); ctx.moveTo(A.x, A.y); ctx.lineTo(B.x, B.y); ctx.stroke();
    });
    k.forEach(p=>{
      if(p.score>=0.3){ ctx.beginPath(); ctx.arc(p.x,p.y,3,0,Math.PI*2); ctx.fill(); }
    });
  }

  // Angle utilities
  function angle(a,b,c){ // angle at b, degrees
    const ab=[a.x-b.x,a.y-b.y], cb=[c.x-b.x,c.y-b.y];
    const dot=ab[0]*cb[0] + ab[1]*cb[1];
    const na=Math.hypot(ab[0],ab[1]), nc=Math.hypot(cb[0],cb[1]);
    const cos = Math.max(-1, Math.min(1, dot/(na*nc+1e-6)));
    return Math.acos(cos)*180/Math.PI;
  }
  function kp(obj, name){ return obj.keypoints.find(p=>p.name===name) || obj.keypoints.find(p=>p.part===name); }

  // Extract a feature vector per frame (angles are scale-invariant)
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
      angle(rh, rk, ra),  // right knee
    ];
  }

  // DTW (L2)
  function dtw(a, b, windowRatio=0.2){
    const n=a.length, m=b.length;
    const w = Math.max(Math.floor(Math.max(n,m)*windowRatio), Math.abs(n-m));
    const INF=1e9;
    const D = Array.from({length:n+1}, ()=>Array(m+1).fill(INF));
    D[0][0]=0;
    function dist(v1,v2){
      let s=0; for(let i=0;i<v1.length;i++){ const d=v1[i]-v2[i]; s+=d*d; }
      return Math.sqrt(s);
    }
    for(let i=1;i<=n;i++){
      const jStart = Math.max(1, i-w), jEnd = Math.min(m, i+w);
      for(let j=jStart;j<=jEnd;j++){
        const cost = dist(a[i-1], b[j-1]);
        D[i][j] = cost + Math.min(D[i-1][j], D[i][j-1], D[i-1][j-1]);
      }
    }
    return D[n][m] / (n+m); // normalized
  }

  // Sample frames → poses → features
  async function sampleSequence(video, canvas, take=90){ // ~3 sec at 30fps if step=1
    await video.play(); video.pause(); // ensure metadata loaded
    const w = canvas.width = video.videoWidth || 854;
    const h = canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');

    const total = Math.floor(video.duration*30); // assume ~30fps
    const step = Math.max(1, Math.floor(total / take)); // uniform samples
    const seq = [];

    for(let f=0; f<total; f+=step){
      video.currentTime = f/30;
      await new Promise(r=> video.onseeked = ()=>r());
      ctx.drawImage(video, 0, 0, w, h);
      const poses = await detector.estimatePoses(canvas, {flipHorizontal:false});
      if(poses && poses[0]){
        drawPose(canvas, poses[0]);
        const feat = extractAngles(poses[0]);
        if(feat) seq.push(feat);
      }
      statusEl.textContent = `Analyzing ${seq.length} frames…`;
    }
    return seq;
  }

  // Turn DTW distance into a 0–100 score
  function scoreFromDist(d){
    // Empirical mapping: 0 distance ~ 100, large distance ~ 0
    const s = 100 * Math.exp(-d/12); // tweak denominator to adjust strictness
    return Math.max(0, Math.min(100, Math.round(s)));
  }

  // Subscores: timing (sequence shape), stance (shoulder/hip), swing (elbows)
  function subScores(seqA, seqB){
    // Split features: [LE, RE, LS, RS, LK, RK]
    const pick = (seq, idxs)=> seq.map(v=> idxs.map(i=>v[i]));
    const elbowA = pick(seqA, [0,1]), elbowB = pick(seqB,[0,1]);
    const shoulderHipA = pick(seqA,[2,3]), shoulderHipB = pick(seqB,[2,3]);
    const kneeA = pick(seqA,[4,5]), kneeB = pick(seqB,[4,5]);
    const dEl = dtw(elbowA, elbowB), dSh = dtw(shoulderHipA, shoulderHipB), dKn = dtw(kneeA, kneeB);
    return {
      swing: scoreFromDist(dEl),
      stance: Math.round((scoreFromDist(dSh)+scoreFromDist(dKn))/2),
      timing: Math.round( ( // approximate timing via whole-body DTW vs per-frame cosine sim
        scoreFromDist(dtw(seqA, seqB)) ) )
    };
  }

  function colorFor(v){ return v>=80?'ok':v>=60?'mid':'bad'; }

  async function analyze(){
    try{
      statusEl.textContent = 'Loading videos…';
      if(proUrl.value) proVid.src = proUrl.value;
      if(!userVid.src){ alert('Please choose your video.'); return; }

      await Promise.all([
        new Promise(r=>proVid.onloadedmetadata=r),
        new Promise(r=>userVid.onloadedmetadata=r)
      ]);

      statusEl.textContent = 'Estimating poses (pro)…';
      const seqPro = await sampleSequence(proVid, proCanvas);

      statusEl.textContent = 'Estimating poses (you)…';
      const seqUser = await sampleSequence(userVid, userCanvas);

      if(seqPro.length<8 || seqUser.length<8){
        statusEl.textContent = 'Not enough frames detected. Try clearer lighting and a single shot clip.';
        return;
      }

      statusEl.textContent = 'Computing DTW…';
      const dist = dtw(seqUser, seqPro, 0.15);
      const mainScore = scoreFromDist(dist);

      const subs = subScores(seqUser, seqPro);

      // UI
      scoreNum.textContent = mainScore;
      scoreBar.style.width = mainScore + '%';
      timingSub.textContent = subs.timing; timingSub.className = colorFor(subs.timing);
      stanceSub.textContent = subs.stance; stanceSub.className = colorFor(subs.stance);
      swingSub.textContent = subs.swing;  swingSub.className = colorFor(subs.swing);

      statusEl.textContent = 'Done ✅';

      // OPTIONAL: wire into your Fun Pack (if present)
      if(window.awardShotResult){
        // Map subs to 0–1
        const t = subs.timing/100, s = subs.stance/100, sw = subs.swing/100;
        window.awardShotResult({ timingScore:t, stanceScore:s, swingScore:sw });
      }
    }catch(e){
      console.error(e);
      statusEl.textContent = 'Error: ' + e.message;
    }
  }

  analyzeBtn.addEventListener('click', analyze);
})();
</script>c
