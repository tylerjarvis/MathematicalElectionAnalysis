import sqlite3
import csv

class ExcelConversion(object):
    """
    Handles the parsing of CSV copied/exported from Excel and the INSERT into a database
    CSV are assumed to be a single race spanning multiple precincts
    """

    # fields in the CSV that are guaranteed and in the same place for every
    # race.
    RegVoterIdx, TimesCountedIdx, TotalVotesIdx = 1, 2, 3

    def __init__(self, filename:str, delimiter:str=None, cursor:sqlite3.Cursor=None):
        """Creates a thing

        @param: filename A CSV/TSV file to read in
        @param: delimiter Optional delimiter to use; if None, it is inferred from the extension
        @param: cursor The database cursor for inserting records. Not currently implemented.        
        """
        self.cursor = cursor
        self.filename = filename
        if delimiter != None:
            self.delimiter = delimiter
        elif filename.endswith(".csv"):
            self.delimiter = ","
        elif filename.endswith(".tsv"):
            self.delimiter = "\t"
        else:
            raise "Delimiter unhandled"


        with open(self.filename, 'r') as fh:
            reader = csv.reader(fh, delimiter= self.delimiter)
            self.lines = [line for line in reader]

        self.Race = None
        self.Precincts = []
        self.Candidates = []

        """aoeu"""
        self.Results = []

    def Parse(self):
        # First cell is the race, eg "US President"
        self.Race = self.lines[0][0]
        
        # Second line is empty

        # Get the candidates. These are on the same line as Registered Voters, etc.
        candidates = [ (i, self.lines[2][i]) for i in range(ExcelConversion.TotalVotesIdx + 1, len(self.lines[2])) if self.lines[2][i].strip() != '']
        self.Candidates = [name for (i,name) in candidates]

        # Precincts are listed as rows starting at index 5 and each take up six rows
        precinct_end = [i for i in range(5, len(self.lines)) if self.lines[i][0].strip() == ''][0]
        precincts = [(i, self.lines[i][0]) for i in range(5, precinct_end, 6) if self.lines[i][0].strip() != '']
        self.Precincts = [name for (i,name) in precincts]

        for (c,candidate) in candidates:
            for (p,precinct) in precincts:
                # add a tuple of (candidate name, precinct name, total votes
                # received)
                # the 'totals' line is offset five from the precinct name
                self.Results.append((candidate, precinct, int(self.lines[p + 5][c])))
        

if __name__ == '__main__':
    ec = ExcelConversion('president.tsv')
    ec.Parse()
    print("Candidates: %s" % ', '.join(ec.Candidates))
    print("Precincts: %s" % ', '.join(ec.Precincts))
