import { serve } from '@hono/node-server'
import { Hono } from 'hono'
import transfertuningRouter from './transfertuning'

const app = new Hono()

app.get('/', (c) => {
    return c.text('Hello Hono!')
})

app.route('/docc', transfertuningRouter)

const server = serve(app, (info) => {
    console.log(`Listening on port ${info.port}`)
})

process.on('SIGINT', () => {
  server.close()
  process.exit(0)
})
process.on('SIGTERM', () => {
  server.close((err) => {
    if (err) {
      console.error(err)
      process.exit(1)
    }
    process.exit(0)
  })
})
