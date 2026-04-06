.. _library-management:

Library Management
==================

Library configuration and management functions for AOCL-DLP.

.. _library-threading-config:

Threading Configuration
-----------------------

Thread Count
~~~~~~~~~~~~
.. _fn-dlp-thread-set-num-threads:
.. doxygenfunction:: dlp_thread_set_num_threads
   :project: aocl-dlp

.. doxygenfunction:: dlp_thread_set_num_threads_library
   :project: aocl-dlp

.. doxygenfunction:: dlp_thread_set_num_threads_local
   :project: aocl-dlp

.. doxygenfunction:: dlp_thread_get_num_threads_active
   :project: aocl-dlp

Thread Ways (2D Decomposition)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: dlp_thread_set_ways
   :project: aocl-dlp

.. doxygenfunction:: dlp_thread_set_ways_library
   :project: aocl-dlp

.. doxygenfunction:: dlp_thread_set_ways_local
   :project: aocl-dlp

.. doxygenfunction:: dlp_thread_get_ic_ways_active
   :project: aocl-dlp

.. doxygenfunction:: dlp_thread_get_jc_ways_active
   :project: aocl-dlp

Hardware Query
--------------

.. doxygenfunction:: dlp_aocl_enable_instruction_query
   :project: aocl-dlp

Version Query
-------------

.. doxygenfunction:: dlp_version_query
   :project: aocl-dlp

.. rubric:: See Also

* :doc:`../gemm/index` - GEMM operations
* :doc:`../gemm/post_ops` - Post-operations framework
* :doc:`../utils/index` - Utility functions
