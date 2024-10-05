

Introduction
------------
Source code of Middleware'24 paper: `Cannikin: Optimal Adaptive Distributed DNN Training over Heterogeneous Clusters <https://arxiv.org/abs/2402.05302>`_

Environment setup
^^^^^^^^^^^^^^^^^^



Easy-to-use Elastic API
^^^^^^^^^^^^^^^^^^^^^^^

Aladdin introduced the HeteroDataLoader for adaptive batch size training over heterogeneous clusters. For other APIs, refer the AdaptDL `Documentation <https://adaptdl.readthedocs.org>`_

**BEFORE:**

.. code-block:: python

   torch.distributed.init_process_group("nccl")
   model = torch.nn.parallel.DistributedDataParallel(model)
   dataloader = torch.utils.data.DataLoader(dataset, batch_size=128)
   for epoch in range(100):
       ...

**AFTER:**

.. code-block:: python

   adaptdl.torch.init_process_group("nccl")
   model = adaptdl.torch.AdaptiveDataParallel(model, optimizer)
   dataloader = adaptdl.torch.HeteroDataLoader(dataset, batch_size=128)
   for epoch in adaptdl.torch.remaining_epochs_until(100):
       ...

.. include-end-before

Getting Started
---------------

Cannikin is built based on the *adaptive training
library* of AdaptDL. It can be used following:


 Adapting the batch size and learning rate for a single training job
    (`Standalone Training <https://adaptdl.readthedocs.io/en/latest/standalone-training.html>`_).

