import { NextRequest, NextResponse } from 'next/server'

const AGENT_API_URL = process.env.AGENT_API_URL || 'http://localhost:8080'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    const response = await fetch(`${AGENT_API_URL}/guardrail/check-target`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        target_domain: body.targetDomain || '',
        target_ips: body.targetIps || [],
        project_id: body.projectId || '',
      }),
    })

    if (!response.ok) {
      // Fail open on agent errors
      return NextResponse.json({ allowed: true, reason: 'Guardrail service unavailable' })
    }

    const data = await response.json()
    return NextResponse.json(data)

  } catch (error) {
    console.error('Guardrail check failed:', error)
    // Fail open
    return NextResponse.json({ allowed: true, reason: 'Guardrail service unreachable' })
  }
}
