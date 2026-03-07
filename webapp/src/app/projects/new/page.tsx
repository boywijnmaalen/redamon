'use client'

import { useRouter } from 'next/navigation'
import { ProjectForm } from '@/components/projects'
import { useCreateProject } from '@/hooks/useProjects'
import { useProject } from '@/providers/ProjectProvider'
import type { Project } from '@prisma/client'
import styles from './page.module.css'

type ProjectFormData = Omit<Project, 'id' | 'userId' | 'createdAt' | 'updatedAt' | 'user'>

export default function NewProjectPage() {
  const router = useRouter()
  const { userId, setCurrentProject } = useProject()
  const createProjectMutation = useCreateProject()

  const handleSubmit = async (data: ProjectFormData) => {
    if (!userId) {
      alert('Please select a user first')
      router.push('/projects')
      return
    }

    try {
      const project = await createProjectMutation.mutateAsync({
        ...data,
        userId,
        name: data.name,
        targetDomain: data.targetDomain
      })

      setCurrentProject({
        id: project.id,
        name: project.name,
        targetDomain: project.targetDomain,
        description: project.description || undefined,
        createdAt: project.createdAt.toString(),
        updatedAt: project.updatedAt.toString()
      })

      router.push(`/graph?project=${project.id}`)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to create project'
      if (message.toLowerCase().includes('guardrail')) {
        throw error // Let ProjectForm handle guardrail errors with its modal
      }
      alert(message)
    }
  }

  const handleCancel = () => {
    router.push('/projects')
  }

  if (!userId) {
    return (
      <div className={styles.container}>
        <div className={styles.message}>
          <p>Please select a user first before creating a project.</p>
          <button className="primaryButton" onClick={() => router.push('/projects')}>
            Go to Projects
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className={styles.container}>
      <ProjectForm
        mode="create"
        onSubmit={handleSubmit}
        onCancel={handleCancel}
        isSubmitting={createProjectMutation.isPending}
      />
    </div>
  )
}
