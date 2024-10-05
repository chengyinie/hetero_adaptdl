

Introduction
------------

`Documentation <https://adaptdl.readthedocs.org>`_ |




Easy-to-use Elastic API
^^^^^^^^^^^^^^^^^^^^^^^

Making training programs run elastically can be challenging and error-prone.
AdaptDL offers APIs which make it easy to enable elasticity for data-parallel
PyTorch programs. Simply change a few lines of code, without heavy refactoring!

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

AdaptDL consists of a *Kubernetes job scheduler* and an *adaptive training
library*. They can be used in two ways:

1.  Scheduling multiple training jobs on a shared cluster or the cloud
    (`Scheduler Installation <https://adaptdl.readthedocs.io/en/latest/installation/index.html>`_).
2.  Adapting the batch size and learning rate for a single training job
    (`Standalone Training <https://adaptdl.readthedocs.io/en/latest/standalone-training.html>`_).

.. image:: _static/img/Petuum.png
  :align: center
