imags = ['/Users/ktompoidi/Documents/tensorFlowDL/data/forms_2cat_1/test/other:test/1617538-3.png',
         '/Users/ktompoidi/Documents/tensorFlowDL/data/forms_2cat_1/test/other:test/1685924-1.png',
         '/Users/ktompoidi/Documents/tensorFlowDL/data/forms_2cat_1/test/other:test/2617020-1.png',
         '/Users/ktompoidi/Documents/tensorFlowDL/data/forms_2cat_1/test/id/1589207-1.png',
         '/Users/ktompoidi/Documents/tensorFlowDL/data/forms_2cat_1/test/id/1593369-1.png',
         '/Users/ktompoidi/Documents/tensorFlowDL/data/forms_2cat_1/test/id/1593672-2.png',
         '/Users/ktompoidi/Documents/tensorFlowDL/data/forms_2cat_1/test/id/1594397-2.png',
         '/Users/ktompoidi/Documents/tensorFlowDL/data/forms_2cat_1/test/id/2590215-1.png',
         '/Users/ktompoidi/Documents/tensorFlowDL/data/forms_2cat_1/test/id/1610823-2.png',
         '/Users/ktompoidi/Documents/tensorFlowDL/data/forms_2cat_1/test/id/1936424-1.png',
         '/Users/ktompoidi/Documents/tensorFlowDL/data/forms_2cat_1/test/id/1689793-1.png',
         '/Users/ktompoidi/Documents/tensorFlowDL/data/forms_2cat_1/test/id/1610823-5.png']

    for image in imags:
        queue.maybe_bind(conn)
        queue.declare()
        producer.publish(image)