"""Example usage of the message queue system."""

import asyncio
import logging
from datetime import timedelta

from hpc_comms.queue.memory_queue import MemoryQueue, QueuePriority
from hpc_comms.queue.distributed_queue import QueueManager, DistributedQueue
from hpc_comms.core.transport import Transport


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExampleWorker:
    """Example worker that processes messages."""
    
    def __init__(self, worker_id: str, queue: MemoryQueue):
        self.worker_id = worker_id
        self.queue = queue
        self.running = False
    
    async def start(self):
        """Start the worker."""
        self.running = True
        logger.info(f"Worker {self.worker_id} started")
        
        while self.running:
            try:
                # Get a message with timeout
                message = await self.queue.get(timeout=1.0)
                
                if message:
                    await self.process_message(message)
                else:
                    # No message available, continue
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                await asyncio.sleep(1.0)
    
    async def process_message(self, message):
        """Process a message."""
        logger.info(f"Worker {self.worker_id} processing message {message.id}")
        
        try:
            # Simulate work
            await asyncio.sleep(0.1)
            
            # Get the payload
            payload = message.payload
            
            # Process based on message type
            if payload.get("type") == "compute":
                result = await self.handle_compute(payload)
            elif payload.get("type") == "data":
                result = await self.handle_data(payload)
            else:
                result = {"status": "unknown_type"}
            
            # Acknowledge successful processing
            await self.queue.ack(message.id)
            logger.info(f"Worker {self.worker_id} completed message {message.id}: {result}")
            
        except Exception as e:
            logger.error(f"Worker {self.worker_id} failed to process message {message.id}: {e}")
            # Negative acknowledge with retry
            await self.queue.nack(message.id, requeue=True)
    
    async def handle_compute(self, payload):
        """Handle compute workload."""
        # Simulate compute work
        work_size = payload.get("size", 1)
        await asyncio.sleep(0.05 * work_size)
        
        return {
            "status": "completed",
            "result": work_size * 2,
            "worker": self.worker_id
        }
    
    async def handle_data(self, payload):
        """Handle data processing."""
        # Simulate data processing
        data_size = payload.get("size", 1)
        await asyncio.sleep(0.02 * data_size)
        
        return {
            "status": "processed",
            "bytes_processed": data_size * 1024,
            "worker": self.worker_id
        }
    
    def stop(self):
        """Stop the worker."""
        self.running = False
        logger.info(f"Worker {self.worker_id} stopped")


async def basic_queue_example():
    """Basic queue usage example."""
    logger.info("=== Basic Queue Example ===")
    
    # Create and start queue
    queue = MemoryQueue(max_size=1000)
    await queue.start()
    
    try:
        # Add some messages with different priorities
        logger.info("Adding messages to queue...")
        
        await queue.put(
            {"type": "compute", "size": 2},
            priority=QueuePriority.HIGH,
            metadata={"source": "urgent"}
        )
        
        await queue.put(
            {"type": "data", "size": 5},
            priority=QueuePriority.NORMAL,
            metadata={"source": "batch"}
        )
        
        await queue.put(
            {"type": "compute", "size": 1},
            priority=QueuePriority.LOW,
            metadata={"source": "background"}
        )
        
        # Show queue stats
        stats = await queue.get_stats()
        logger.info(f"Queue stats: {stats.to_dict()}")
        
        # Process messages
        logger.info("Processing messages...")
        
        for i in range(3):
            message = await queue.get(timeout=1.0)
            if message:
                logger.info(f"Got message: {message.id}, priority: {message.priority}")
                logger.info(f"Payload: {message.payload}")
                
                # Acknowledge
                await queue.ack(message.id)
            else:
                logger.info("No message available")
        
        # Final stats
        stats = await queue.get_stats()
        logger.info(f"Final queue stats: {stats.to_dict()}")
        
    finally:
        await queue.stop()


async def worker_pool_example():
    """Example with multiple workers."""
    logger.info("=== Worker Pool Example ===")
    
    # Create and start queue
    queue = MemoryQueue(max_size=1000)
    await queue.start()
    
    # Create workers
    workers = [
        ExampleWorker(f"worker_{i}", queue)
        for i in range(3)
    ]
    
    try:
        # Start workers
        worker_tasks = [
            asyncio.create_task(worker.start())
            for worker in workers
        ]
        
        # Add work items
        logger.info("Adding work items...")
        
        for i in range(10):
            priority = QueuePriority.HIGH if i % 3 == 0 else QueuePriority.NORMAL
            
            await queue.put(
                {
                    "type": "compute" if i % 2 == 0 else "data",
                    "size": (i % 5) + 1,
                    "job_id": i
                },
                priority=priority,
                metadata={"batch": "example_batch"}
            )
        
        # Let workers process for a while
        await asyncio.sleep(2.0)
        
        # Show stats
        stats = await queue.get_stats()
        logger.info(f"Queue stats after processing: {stats.to_dict()}")
        
        # Stop workers
        for worker in workers:
            worker.stop()
        
        # Wait for workers to finish
        await asyncio.gather(*worker_tasks, return_exceptions=True)
        
    finally:
        await queue.stop()


async def distributed_queue_example():
    """Example with distributed queue."""
    logger.info("=== Distributed Queue Example ===")
    
    # Mock transport for demonstration
    class MockTransport:
        async def send_message(self, endpoint: str, message):
            logger.info(f"Sending message to {endpoint}: {message.message_type}")
            await asyncio.sleep(0.01)  # Simulate network delay
    
    # Create queue manager
    transport = MockTransport()
    manager = QueueManager(
        transport=transport,
        node_id="node_1",
        replication_factor=2
    )
    
    await manager.start()
    
    try:
        # Add some nodes to the cluster
        await manager.add_node("node_2", "http://node2:8080")
        await manager.add_node("node_3", "http://node3:8080")
        
        # Create distributed queue
        queue = DistributedQueue(manager)
        
        # Add work items
        logger.info("Adding distributed work items...")
        
        for i in range(5):
            await queue.put(
                {
                    "type": "compute",
                    "size": (i % 3) + 1,
                    "job_id": f"distributed_job_{i}"
                },
                priority=QueuePriority.NORMAL,
                metadata={"distributed": True}
            )
        
        # Process some items
        logger.info("Processing distributed work items...")
        
        for i in range(3):
            message = await queue.get(timeout=1.0)
            if message:
                logger.info(f"Processing distributed message: {message.id}")
                await asyncio.sleep(0.05)  # Simulate work
                await queue.ack(message.id)
        
        # Show cluster stats
        stats = await queue.get_stats()
        logger.info(f"Distributed queue stats: {stats.to_dict()}")
        
    finally:
        await manager.stop()


async def priority_and_retry_example():
    """Example demonstrating priority and retry behavior."""
    logger.info("=== Priority and Retry Example ===")
    
    # Create queue
    queue = MemoryQueue(max_size=100)
    await queue.start()
    
    try:
        # Add messages with different priorities
        await queue.put({"task": "low_priority"}, priority=QueuePriority.LOW)
        await queue.put({"task": "high_priority"}, priority=QueuePriority.HIGH)
        await queue.put({"task": "normal_priority"}, priority=QueuePriority.NORMAL)
        
        # Add a message that will fail and retry
        await queue.put(
            {"task": "failing_task"},
            max_retries=2,
            metadata={"will_fail": True}
        )
        
        # Process messages
        logger.info("Processing messages with priority...")
        
        for i in range(4):
            message = await queue.get(timeout=1.0)
            if message:
                logger.info(f"Message {i+1}: {message.payload} (priority: {message.priority.name})")
                
                # Simulate failure for the failing task
                if message.payload.get("task") == "failing_task":
                    logger.info(f"Failing task {message.id}, retry count: {message.retry_count}")
                    await queue.nack(message.id, requeue=True)
                else:
                    await queue.ack(message.id)
        
        # Check for dead letter messages
        dead_messages = await queue.get_dead_letter_messages()
        if dead_messages:
            logger.info(f"Dead letter messages: {len(dead_messages)}")
            for msg in dead_messages:
                logger.info(f"  - {msg.id}: {msg.payload} (retries: {msg.retry_count})")
        
    finally:
        await queue.stop()


async def main():
    """Run all examples."""
    logger.info("Starting queue examples...")
    
    await basic_queue_example()
    await asyncio.sleep(0.5)
    
    await worker_pool_example()
    await asyncio.sleep(0.5)
    
    await distributed_queue_example()
    await asyncio.sleep(0.5)
    
    await priority_and_retry_example()
    
    logger.info("Queue examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
