import { Hono } from 'hono'


const router = new Hono()

router.post('/get-recipe', async (c) => {
    // Placeholder implementation
    return c.json({ data: { sequence: matmul_sequence, region_id: "123", speedup: 1.5, vector_distance: 0.12 } })
})

export default router

const matmul_sequence = [
    {
        "parameters": {
            "tile_size": 32
        },
        "subgraph": {
            "0": {
                "element_id": 1,
                "type": "map"
            }
        },
        "transformation_type": "LoopTiling"
    },
    {
        "parameters": {
            "tile_size": 32
        },
        "subgraph": {
            "0": {
                "element_id": 4,
                "type": "for"
            }
        },
        "transformation_type": "LoopTiling"
    },
    {
        "parameters": {
            "tile_size": 32
        },
        "subgraph": {
            "0": {
                "element_id": 7,
                "type": "map"
            }
        },
        "transformation_type": "LoopTiling"
    },
    {
        "subgraph": {
            "0": {
                "element_id": 1,
                "type": "map"
            },
            "1": {
                "element_id": 24,
                "type": "for"
            }
        },
        "transformation_type": "LoopInterchange"
    },
    {
        "subgraph": {
            "0": {
                "element_id": 4,
                "type": "for"
            },
            "1": {
                "element_id": 27,
                "type": "map"
            }
        },
        "transformation_type": "LoopInterchange"
    },
    {
        "subgraph": {
            "0": {
                "element_id": 33,
                "type": "map"
            },
            "1": {
                "element_id": 36,
                "type": "map"
            }
        },
        "transformation_type": "LoopInterchange"
    }
]
