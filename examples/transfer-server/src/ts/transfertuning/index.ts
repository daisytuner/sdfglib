import { Hono } from 'hono'
import { promises as fs } from 'fs'


const router = new Hono()

router.post('/get-recipe', async (c) => {
    // Optional testing mode: if paths are provided via headers, load JSONs from disk.
    const sdfgPath = c.req.header('sdfg-path')
    const sequencePath = c.req.header('sequence-path')

    if (sdfgPath && sequencePath) {
        try {
            const [sdfgRaw, sequenceRaw] = await Promise.all([
                fs.readFile(sdfgPath, 'utf-8'),
                fs.readFile(sequencePath, 'utf-8'),
            ])

            const sdfgJson = JSON.parse(sdfgRaw)
            const sequenceJson = JSON.parse(sequenceRaw)

            return c.json({
                data: [
                    {
                        sdfg: sdfgJson,
                        sequence: sequenceJson,
                        region_id: 'test-region',
                        speedup: 1.0,
                        vector_distance: 0.0,
                    },
                ],
            })
        } catch (err) {
            console.error('Failed to load testing JSON files:', err)
            return c.json({ error: 'Failed to load testing JSON files' }, 500)
        }
    }

    // Default behaviour: fall back to hard-coded matmul example.
    return c.json({ data: [{ sdfg: "", sequence: [], region_id: "123", speedup: 1.0, vector_distance: 1.0 }] })
})

export default router
