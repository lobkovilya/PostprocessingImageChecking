import exifread

def analyze_metadata(filename):
        time1 = 0
        time2 = 0
        level_of_distrust = 0
        image = open(filename, 'rb')
        tags = exifread.process_file(image)
        for tag in tags.keys():
                print ("%s: %s" % (tag, tags[tag]))
                if tag in 'Image Software':
                    level_of_distrust += 1
                if tag in 'Image DateTime':
                    time1 = tags[tag]
                if tag in 'EXIF DateTimeDigitized':
                    time2 = tags[tag]

        if(time1 and time2 and time1 != time2):
            level_of_distrust += 1

        print ("\nLevel of distrust of metadata (from 0 to 2): %d" % level_of_distrust)

analyze_metadata("insertion.jpg")