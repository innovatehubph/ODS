import {useCallback, useEffect, useState} from 'react'
import {Loader2, RefreshCw} from 'lucide-react'
import HelpLink from '../HelpLink'

const THINKING_TEXT = {
  enable_thinking: 'Thinking can be turned off',
  always: 'Always thinks before it answers',
  none: 'Answers without a thinking step',
}

async function readJson(response) {
  try { return await response.json() } catch { return null }
}

function chips(summary) {
  const list = []
  if (summary.chat === true) list.push({tone: 'ok', text: 'Answers chat'})
  if (summary.chat === false) list.push({tone: 'warn', text: 'Did not answer the chat check'})
  if (summary.tools === true) list.push({tone: 'ok', text: 'Calls tools (Pixel and agents)'})
  if (summary.tools === false) list.push({tone: 'warn', text: 'No working tool calls: chat only for agents'})
  const thinking = THINKING_TEXT[summary.thinking?.control]
  if (thinking) list.push({tone: 'neutral', text: thinking})
  if (summary.vision === true) list.push({tone: 'ok', text: 'Sees images'})
  if (summary.vision === false) list.push({tone: 'warn', text: 'Image input did not work'})
  if (typeof summary.tokensPerSecond === 'number') list.push({tone: 'neutral', text: `About ${Math.round(summary.tokensPerSecond)} tokens/s`})
  return list
}

const TONES = {
  ok: 'border-emerald-300/25 bg-emerald-400/10 text-emerald-200',
  warn: 'border-amber-300/25 bg-amber-400/10 text-amber-100',
  neutral: 'border-theme-border bg-theme-text-secondary/10 text-theme-text-secondary',
}

/** Advisory: what the running model was measured to do on this machine. */
export default function ModelProfileSummary({modelId}) {
  const [data, setData] = useState(null)
  const [failed, setFailed] = useState(false)
  const [rechecking, setRechecking] = useState(false)
  const [notice, setNotice] = useState('')

  const load = useCallback(async () => {
    try {
      const response = await fetch(`/api/models/${encodeURIComponent(modelId)}/profile`, {cache: 'no-store'})
      const body = await readJson(response)
      if (!response.ok || !body || typeof body.mode !== 'string') throw new Error('unavailable')
      setData(body)
      setFailed(false)
    } catch {
      setFailed(true)
    }
  }, [modelId])

  useEffect(() => {
    if (modelId) load()
  }, [modelId, load])

  const recheck = async () => {
    setRechecking(true)
    setNotice('')
    try {
      const response = await fetch(`/api/models/${encodeURIComponent(modelId)}/profile/recheck`, {method: 'POST'})
      const body = await readJson(response)
      if (!response.ok) {
        const detail = body?.detail
        setNotice(typeof detail === 'string' ? detail : detail?.message || detail?.error || 'The check could not run. Try again in a minute.')
      } else if (body?.status === 'error') {
        setNotice('The check did not finish. Chat still works; try again later.')
      }
      await load()
    } finally {
      setRechecking(false)
    }
  }

  if (!modelId || failed || !data || data.mode === 'off') return null
  const recheckButton = (
    <button
      type="button"
      onClick={recheck}
      disabled={rechecking}
      className="inline-flex items-center gap-1 rounded border border-theme-border px-2 py-0.5 text-theme-text-secondary hover:text-theme-text disabled:opacity-50"
    >
      {rechecking ? <Loader2 size={11} className="animate-spin" /> : <RefreshCw size={11} />}
      {rechecking ? 'Checking…' : 'Check again'}
    </button>
  )
  const profile = data.profile
  if (!profile) {
    // A model running since before checks existed has no profile yet. The
    // Portal advisory sends the owner here to press Check again.
    return (
      <div className="mt-2 space-y-1.5 text-[11px] text-theme-text-muted">
        <p>Not checked yet: ODS checks what a model can do the first time it runs on this machine.</p>
        {recheckButton}
        {notice && <p role="status" className="text-amber-200">{notice}</p>}
      </div>
    )
  }
  const summary = profile.result?.summary || {}
  const facts = profile.result?.facts || {}
  const list = chips(summary)
  const hasWarning = list.some(chip => chip.tone === 'warn')
  const checkedAt = Date.parse(profile.recordedAt || '')
  return (
    <section aria-label="What this model can do" className="mt-3 space-y-2 text-[11px]">
      <div className="flex flex-wrap gap-1.5">
        {list.map(chip => (
          <span key={chip.text} className={`rounded-md border px-2 py-0.5 ${TONES[chip.tone]}`}>{chip.text}</span>
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-theme-text-muted">
        <span>
          Checked {Number.isFinite(checkedAt) ? new Date(checkedAt).toLocaleDateString() : 'earlier'}
          {facts.buildInfo ? ` on llama.cpp ${String(facts.buildInfo).split('-')[0]}` : ''}
          {profile.result?.status === 'partial' ? '; some checks ran out of time' : ''}.
        </span>
        {recheckButton}
        {hasWarning && <HelpLink className="text-[11px]" />}
      </div>
      {notice && <p role="status" className="text-amber-200">{notice}</p>}
    </section>
  )
}
