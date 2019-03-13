import sqlite3
import csv


class RaceConversion(object):
    """
    Handles the parsing of CSV copied/exported from Excel and the INSERT into a database
    CSV are assumed to be a single race spanning multiple precincts
    """

    # fields in the CSV that are guaranteed and in the same place for every
    # race.
    RegVoterIdx, TimesCountedIdx, TotalVotesIdx = 1, 2, 3

    def __init__(self, lines):
        self.lines = lines

        self.Race = None
        self.Precincts = []
        self.Candidates = []
        self.Results = []

    def Parse(self):
        # First cell is the race, eg "US President"
        self.Race = self.lines[0][0]
        
        # Second line is empty

        # Get the candidates.  These are on the same line as Registered
        # Voters, etc.
        candidates = [ (i, self.lines[2][i]) for i in range(RaceConversion.TotalVotesIdx + 1, len(self.lines[2])) if self.lines[2][i].strip() != '']
        self.Candidates = [name for (i,name) in candidates]

        # Precincts are listed as rows starting at index 5 and each take up
        # six rows
        precinct_end = [i for i in range(5, len(self.lines)) if self.lines[i][0].strip() == ''][0]
        precincts = [(i, self.lines[i][0]) for i in range(5, precinct_end, 6) if self.lines[i][0].strip() != '']
        self.Precincts = [name for (i,name) in precincts]

        for (p,precinct) in precincts:
            # Vote counts for this precinct. The 'totals' line is offset five from the precinct name.
            precinct_votes = [cell for cell in self.lines[p + 5]]

            # Check if this precinct lacks information for this candidate. If so, skip it. First column is "Total"
            if all([cell.strip() == '' for cell in precinct_votes[1:]]):
                continue

            for (c,candidate) in candidates:
                # Add a tuple of (candidate name, precinct name, #votes received)
                self.Results.append((candidate, precinct, int(precinct_votes[c])))
 
class ExcelConversion(object):
    def __init__(self, filename:str, delimiter:str=None, cursor:sqlite3.Cursor=None):
        """
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

        self.Races = []

    def Parse(self):
        """Split up the parsed CSV into child lists and hand it off to RaceConversion
        """
        # Find the line with the race description
        index = [i for i in range(len(self.lines)) if self.lines[i][0] == 'TURN OUT'][0]
        subset = self.lines[index:]

        # get the names and indexes of each race
        races = [(i, self.lines[index][i]) for i in range(len(self.lines[index])) if self.lines[index][i].strip() != '']

        # dummy record
        races.append((None, None))

        for i in range(len(races) - 1):
            (current_index, current_name) = races[i]
            (next_index, next_name) = races[i + 1]

            if current_name in ('TURN OUT', 'STRAIGHT PARTY'):
                continue

            # slice the list[][] and create a RaceConversion
            raceLines = [line[current_index:next_index] for line in subset]
            race = RaceConversion(raceLines)
            race.Parse()

            self.Races.append(race)
       

if __name__ == '__main__':
    ec = ExcelConversion('beaver.2008.SOVC (1).tsv')
    ec.Parse()

    print("Found %d races" % len(ec.Races))
