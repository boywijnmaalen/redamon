import { NextRequest, NextResponse } from 'next/server'
import prisma from '@/lib/prisma'

const AGENT_API_URL = process.env.AGENT_API_URL || 'http://localhost:8080'

// GET /api/projects - List projects (optional user_id filter)
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const userId = searchParams.get('userId')

    const projects = await prisma.project.findMany({
      where: userId ? { userId } : undefined,
      orderBy: { createdAt: 'desc' },
      select: {
        id: true,
        userId: true,
        name: true,
        description: true,
        targetDomain: true,
        createdAt: true,
        updatedAt: true,
        user: {
          select: {
            id: true,
            name: true,
            email: true
          }
        }
      }
    })

    return NextResponse.json(projects)
  } catch (error) {
    console.error('Failed to fetch projects:', error)
    return NextResponse.json(
      { error: 'Failed to fetch projects' },
      { status: 500 }
    )
  }
}

// POST /api/projects - Create a new project
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { userId, name, targetDomain, ipMode, ...optionalParams } = body

    if (!userId || !name) {
      return NextResponse.json(
        { error: 'userId and name are required' },
        { status: 400 }
      )
    }

    // targetDomain is required only when not in IP mode
    if (!ipMode && !targetDomain) {
      return NextResponse.json(
        { error: 'targetDomain is required when not in IP mode' },
        { status: 400 }
      )
    }

    // Verify user exists
    const user = await prisma.user.findUnique({ where: { id: userId } })
    if (!user) {
      return NextResponse.json(
        { error: 'User not found' },
        { status: 404 }
      )
    }

    // Target guardrail: check if domain/IPs are allowed before creating
    try {
      const guardrailResponse = await fetch(`${AGENT_API_URL}/guardrail/check-target`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_domain: ipMode ? '' : (targetDomain || ''),
          target_ips: ipMode ? (optionalParams.targetIps || []) : [],
        }),
      })

      if (guardrailResponse.ok) {
        const guardrailResult = await guardrailResponse.json()
        if (guardrailResult.allowed === false) {
          return NextResponse.json(
            { error: `Target blocked by guardrail: ${guardrailResult.reason}` },
            { status: 403 }
          )
        }
      }
      // If guardrail is unreachable or returns non-OK, fail open (allow)
    } catch (guardrailError) {
      console.warn('Guardrail check failed, proceeding with project creation:', guardrailError)
    }

    // Create project with required fields and any optional params
    const project = await prisma.project.create({
      data: {
        userId,
        name,
        targetDomain: ipMode ? '' : targetDomain,
        ipMode: ipMode || false,
        ...optionalParams
      }
    })

    return NextResponse.json(project, { status: 201 })
  } catch (error) {
    console.error('Failed to create project:', error)
    return NextResponse.json(
      { error: 'Failed to create project' },
      { status: 500 }
    )
  }
}
