import { serve } from '@hono/node-server'
import { Hono } from 'hono'
import transfertuningRouter from './transfertuning'
import { logger as honoLogger } from 'hono/logger'

const app = new Hono()

app.use(honoLogger())

app.get('/', (c) => {
    return c.text('SDFGLib Demo Transfertuning Server is running')
})

app.route('/docc', transfertuningRouter)

const server = serve({
    fetch: app.fetch,
    port: 8080
}, (info) => {
    console.log(`Listening on port ${info.port}`)
})

process.on('SIGINT', () => {
    console.log('SIGINT: Shutting down server...')
    server.close()
    process.exit(0)
})
process.on('SIGTERM', () => {
    console.log('Shutting down server...')
    server.close((err) => {
        if (err) {
        console.error(err)
        process.exit(1)
        }
        process.exit(0)
    })
})
