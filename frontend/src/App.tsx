import { ChangeEvent, DragEvent, useCallback, useEffect, useRef, useState } from 'react'

type Prediction = {
  rank: number
  scientific_name: string
  species_id: string
  score: number
}

type PredictionResponse = {
  model: string
  predictions: Prediction[]
  inference_time_ms: number
}

type Mode = 'upload' | 'camera'

const API_URL = import.meta.env.VITE_API_URL ?? '/api'
const MAX_FILE_SIZE = 10 * 1024 * 1024
const AUTO_SCAN_MS = 8000

function LeafLogo() {
  return (
    <svg viewBox="0 0 48 48" aria-hidden="true">
      <path d="M38.9 7.8C26.6 7.6 15.7 11.6 10.6 21c-3.9 7.1-.5 14.4 5.1 17.7 1.6-7.5 6.8-14 15-18.7-6.5 5.8-10.4 12-11.5 18.9 6.6.7 12.9-2.4 15.8-8.2 3.1-6.1 2.2-14.2 3.9-22.9Z" />
    </svg>
  )
}

function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4" />
    </svg>
  )
}

function CameraIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 7.5h3l1.4-2h7.2l1.4 2h3a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1v-10a1 1 0 0 1 1-1Z" />
      <circle cx="12" cy="13" r="3.5" />
    </svg>
  )
}

export default function App() {
  const [mode, setMode] = useState<Mode>('upload')
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const [cameraActive, setCameraActive] = useState(false)
  const [autoScan, setAutoScan] = useState(true)
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const requestActiveRef = useRef(false)

  const clearResult = useCallback(() => {
    setResult(null)
    setError(null)
  }, [])

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setCameraActive(false)
  }, [])

  useEffect(() => () => stopCamera(), [stopCamera])

  const analyzeBlob = useCallback(async (blob: Blob, filename: string, imagePreview?: string) => {
    if (requestActiveRef.current) return
    requestActiveRef.current = true
    setLoading(true)
    setError(null)
    if (imagePreview) setPreview(imagePreview)

    const data = new FormData()
    data.append('file', blob, filename)
    try {
      const response = await fetch(`${API_URL}/predict`, { method: 'POST', body: data })
      const body = await response.json().catch(() => null)
      if (!response.ok) {
        if (response.status >= 500 && !body?.error?.message) {
          throw new Error('The recognition API is offline. Start the FastAPI server on port 8000.')
        }
        throw new Error(body?.error?.message ?? `Recognition failed with HTTP ${response.status}.`)
      }
      setResult(body as PredictionResponse)
    } catch (requestError) {
      setError(
        requestError instanceof TypeError
          ? 'The recognition service is unavailable. Make sure the API is running.'
          : requestError instanceof Error
            ? requestError.message
            : 'Something went wrong.',
      )
    } finally {
      requestActiveRef.current = false
      setLoading(false)
    }
  }, [])

  const acceptFile = useCallback((nextFile: File) => {
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(nextFile.type)) {
      setError('Choose a JPG, PNG, or WebP image.')
      return
    }
    if (nextFile.size > MAX_FILE_SIZE) {
      setError('Choose an image smaller than 10 MB.')
      return
    }
    if (preview?.startsWith('blob:')) URL.revokeObjectURL(preview)
    setFile(nextFile)
    setPreview(URL.createObjectURL(nextFile))
    clearResult()
  }, [clearResult, preview])

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0]
    if (nextFile) acceptFile(nextFile)
  }

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragging(false)
    const nextFile = event.dataTransfer.files[0]
    if (nextFile) acceptFile(nextFile)
  }

  const reset = () => {
    if (preview?.startsWith('blob:')) URL.revokeObjectURL(preview)
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  const switchMode = (nextMode: Mode) => {
    if (nextMode === mode) return
    if (mode === 'camera') stopCamera()
    reset()
    setMode(nextMode)
  }

  const startCamera = async () => {
    clearResult()
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera access is not supported in this browser or context.')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 } },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      setCameraActive(true)
    } catch {
      setError('Camera permission was denied or no camera is available.')
    }
  }

  const captureAndAnalyze = useCallback(() => {
    const video = videoRef.current
    if (!video || video.readyState < 2 || requestActiveRef.current) return
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const context = canvas.getContext('2d')
    if (!context) return
    context.drawImage(video, 0, 0)
    canvas.toBlob((blob) => {
      if (!blob) return
      analyzeBlob(blob, 'camera-capture.jpg', canvas.toDataURL('image/jpeg', 0.86))
    }, 'image/jpeg', 0.9)
  }, [analyzeBlob])

  useEffect(() => {
    if (!cameraActive || !autoScan) return
    const firstScan = window.setTimeout(captureAndAnalyze, 1200)
    const interval = window.setInterval(captureAndAnalyze, AUTO_SCAN_MS)
    return () => {
      window.clearTimeout(firstScan)
      window.clearInterval(interval)
    }
  }, [cameraActive, autoScan, captureAndAnalyze])

  const topPrediction = result?.predictions[0]

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Flora Lens AI home">
          <span className="brand-mark"><LeafLogo /></span>
          <span>Flora Lens <b>AI</b></span>
        </a>
        <div className="model-chip"><span /> PlantCLEF 2024</div>
      </header>

      <main id="top">
        <section className="hero">
          <div className="eyebrow"><span>✦</span> Discover nature through AI</div>
          <h1>Every leaf has a name.<br /><em>Let AI find it.</em></h1>
          <p>
            Identify plants from a photo in seconds, powered by a specialist computer vision
            model trained across 7,806 species.
          </p>
          <div className="trust-row">
            <span><i>✓</i> Real model inference</span>
            <span><i>✓</i> Private by design</span>
            <span><i>✓</i> No sign-up</span>
          </div>
        </section>

        <section className="recognizer" aria-label="Plant recognition workspace">
          <div className="mode-tabs" role="tablist" aria-label="Image source">
            <button className={mode === 'upload' ? 'active' : ''} onClick={() => switchMode('upload')} role="tab" aria-selected={mode === 'upload'}>
              <UploadIcon /> Upload photo
            </button>
            <button className={mode === 'camera' ? 'active' : ''} onClick={() => switchMode('camera')} role="tab" aria-selected={mode === 'camera'}>
              <CameraIcon /> Use camera
            </button>
          </div>

          <div className="workspace-grid">
            <div className="input-panel">
              {mode === 'upload' ? (
                <>
                  {!file ? (
                    <div
                      className={`drop-zone ${dragging ? 'dragging' : ''}`}
                      onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
                      onDragLeave={() => setDragging(false)}
                      onDrop={onDrop}
                      onClick={() => inputRef.current?.click()}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(event) => event.key === 'Enter' && inputRef.current?.click()}
                    >
                      <div className="upload-orbit"><UploadIcon /></div>
                      <h2>Drop your plant photo here</h2>
                      <p>or click to browse your device</p>
                      <button className="secondary-button" type="button">Choose an image</button>
                      <small>JPG, PNG or WebP · max 10 MB</small>
                    </div>
                  ) : (
                    <div className="image-stage">
                      <img src={preview ?? ''} alt="Plant selected for recognition" />
                      <div className="stage-overlay">
                        <span>{file.name}</span>
                        <button onClick={reset} aria-label="Remove selected image">×</button>
                      </div>
                      {loading && <div className="scan-line" />}
                    </div>
                  )}
                  <input ref={inputRef} className="sr-only" type="file" accept="image/jpeg,image/png,image/webp" onChange={onFileChange} />
                  <button
                    className="primary-button"
                    disabled={!file || loading}
                    onClick={() => file && analyzeBlob(file, file.name)}
                  >
                    {loading ? <><span className="spinner" /> Reading botanical patterns…</> : <>Identify this plant <span>→</span></>}
                  </button>
                </>
              ) : (
                <div className="camera-wrap">
                  <div className={`camera-stage ${cameraActive ? 'live' : ''}`}>
                    <video ref={videoRef} muted playsInline aria-label="Live camera preview" />
                    {!cameraActive && (
                      <div className="camera-empty">
                        <div className="upload-orbit"><CameraIcon /></div>
                        <h2>Point your camera at a plant</h2>
                        <p>Use a clear, well-lit view of leaves or flowers.</p>
                      </div>
                    )}
                    {cameraActive && <div className="focus-frame"><i /><i /><i /><i /></div>}
                    {cameraActive && <span className="live-badge"><b /> Live</span>}
                    {loading && <div className="scan-line" />}
                  </div>
                  <div className="camera-controls">
                    {!cameraActive ? (
                      <button className="primary-button" onClick={startCamera}><CameraIcon /> Start camera</button>
                    ) : (
                      <>
                        <button className="primary-button" onClick={captureAndAnalyze} disabled={loading}>Scan now</button>
                        <button className="text-button" onClick={stopCamera}>Stop camera</button>
                      </>
                    )}
                  </div>
                  {cameraActive && (
                    <label className="auto-toggle">
                      <input type="checkbox" checked={autoScan} onChange={(event) => setAutoScan(event.target.checked)} />
                      <span /> Auto-identify every 8 seconds
                    </label>
                  )}
                </div>
              )}
              {error && <div className="error-message" role="alert">{error}</div>}
            </div>

            <aside className={`results-panel ${result ? 'has-results' : ''}`} aria-live="polite">
              {!result ? (
                <div className="empty-result">
                  <div className="specimen-icon"><LeafLogo /><span>01</span></div>
                  <h2>Your result will appear here</h2>
                  <p>We’ll compare visual patterns against thousands of plant species.</p>
                  <div className="empty-lines"><span /><span /><span /></div>
                </div>
              ) : (
                <div className="result-content">
                  <div className="result-heading">
                    <span className="result-label">Best match</span>
                    <span className="timing">{result.inference_time_ms.toLocaleString()} ms</span>
                  </div>
                  <h2>{topPrediction?.scientific_name}</h2>
                  <p className="species-id">PlantNet species ID · {topPrediction?.species_id}</p>
                  <div className="confidence-feature">
                    <div>
                      <strong>{((topPrediction?.score ?? 0) * 100).toFixed(1)}%</strong>
                      <span>model confidence</span>
                    </div>
                    <svg viewBox="0 0 42 42" aria-hidden="true">
                      <circle cx="21" cy="21" r="16" />
                      <circle cx="21" cy="21" r="16" pathLength="100" style={{ strokeDasharray: `${(topPrediction?.score ?? 0) * 100} 100` }} />
                    </svg>
                  </div>
                  <div className="ranking-title"><span>Top matches</span><small>Confidence</small></div>
                  <ol className="ranking-list">
                    {result.predictions.map((prediction) => (
                      <li key={`${prediction.rank}-${prediction.species_id}`}>
                        <span className="rank">{String(prediction.rank).padStart(2, '0')}</span>
                        <div className="prediction-name">
                          <strong>{prediction.scientific_name}</strong>
                          <div><i style={{ width: `${Math.max(prediction.score * 100, 2)}%` }} /></div>
                        </div>
                        <span className="score">{(prediction.score * 100).toFixed(1)}%</span>
                      </li>
                    ))}
                  </ol>
                  <button className="text-button new-scan" onClick={reset}>Identify another plant →</button>
                </div>
              )}
            </aside>
          </div>
        </section>

        <section className="how-it-works">
          <div><span>01</span><h3>Choose a clear photo</h3><p>Frame one plant with visible leaves or flowers.</p></div>
          <div><span>02</span><h3>AI studies the details</h3><p>DINOv2 analyzes shape, color, texture, and structure.</p></div>
          <div><span>03</span><h3>Explore the matches</h3><p>Review the five most likely scientific names.</p></div>
        </section>
      </main>

      <footer>
        <div className="brand"><span className="brand-mark"><LeafLogo /></span><span>Flora Lens <b>AI</b></span></div>
        <p>Identification is an AI estimate and may be incorrect. Do not use it to decide whether a plant is safe to eat or touch.</p>
      </footer>
    </div>
  )
}

