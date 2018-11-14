import os
import sqlite3
import json

import pandas as pd
import psycopg2

from clefts.catmaid_interface import get_catmaid
from clefts.constants import (
    CONN_CACHE_PATH,
    DB_CREDENTIALS_PATH,
    BASIN_CACHE_PATH,
    BASIN_ANNOTATION,
    CoordZYX,
    PROJECT_ID,
)

with open(DB_CREDENTIALS_PATH) as f:
    db_creds = json.load(f)

conn = psycopg2.connect(**db_creds)
cursor = conn.cursor()


def get_caches(conn_path=CONN_CACHE_PATH, basin_path=BASIN_CACHE_PATH, force=False):
    should_populate_conns = force or not os.path.exists(conn_path)
    all_cache = ConnectorCache(conn_path, force)
    if should_populate_conns:
        all_cache.populate_from(conn)

    should_populate_basin = force or not os.path.exists(basin_path)
    basin_cache = ConnectorCache(basin_path, force)
    if should_populate_basin:
        catmaid = get_catmaid()
        basin_skids = [
            row["skeleton_id"]
            for row in catmaid.get_skeletons_by_annotation(BASIN_ANNOTATION)
        ]
        basin_cids = catmaid.get_synapses_among(basin_skids)["connector_id"]
        basin_cache.populate_from(conn, basin_cids)

    return all_cache, basin_cache


class ConnectorCache:
    def __init__(self, cache_path, recreate=False):
        self.cache_path = cache_path
        self.conn = self._get_connection(recreate)

    def __enter__(self):
        return self.conn.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.conn.__exit__(exc_type, exc_val, exc_tb)

    def __len__(self):
        return self.cursor().execute("SELECT count(*) FROM connector;").fetchone()[0]

    def iter_raw(self):
        yield from self.cursor().execute("SELECT * FROM connector;")

    def __iter__(self):
        for row in self.iter_raw():
            yield row[0], CoordZYX(row[1:])

    def __getitem__(self, item):
        row = (
            self.cursor()
            .execute("SELECT * FROM connector WHERE id = ? LIMIT 1", (item,))
            .fetchone()
        )
        return CoordZYX(row[1:])

    def cursor(self, *args, **kwargs):
        return self.conn.cursor(*args, **kwargs)

    def _get_connection(self, recreate):
        if recreate:
            try:
                os.remove(self.cache_path)
            except OSError:
                pass

        should_setup = not os.path.exists(self.cache_path)
        conn = sqlite3.connect(self.cache_path)

        if should_setup:
            cache_cursor = conn.cursor()
            cache_cursor.execute(
                "CREATE TABLE connector (id INTEGER PRIMARY KEY, z REAL, y REAL, x REAL);"
            )
            cache_cursor.execute("CREATE INDEX z_idx ON connector (z);")
            cache_cursor.execute("CREATE INDEX y_idx ON connector (y);")
            cache_cursor.execute("CREATE INDEX x_idx ON connector (x);")

        return conn

    def populate_from(self, connection=conn, id_set=None, project_id=PROJECT_ID):
        src_curs = connection.cursor()
        if id_set is None:
            src_curs.execute(
                "SELECT id, location_z, location_y, location_x FROM connector WHERE project_id = %s;",
                (project_id,),
            )
        else:
            src_curs.execute(
                """
                SELECT c.id, c.location_z, c.location_y, c.location_x FROM connector c
                  INNER JOIN unnest(%(obj_ids)s::BIGINT[]) AS syns (id)
                  ON c.id = syns.id
                  WHERE project_id = %(project_id)s;
            """,
                {"obj_ids": list(id_set), "project_id": project_id},
            )

        with self:
            self.cursor().executemany(
                "INSERT OR IGNORE INTO connector (id, z, y, x) VALUES (?, ?, ?, ?)",
                src_curs.fetchall(),
            )

    def _roi_to_params(self, offset, shape):
        upper = offset + shape
        return {
            "zmin": offset.z,
            "zmax": upper.z,
            "ymin": offset.y,
            "ymax": upper.y,
            "xmin": offset.x,
            "xmax": upper.x,
        }

    def count_in_box(self, offset, shape):
        query = """
          SELECT count(*) FROM connector 
            WHERE z BETWEEN :zmin AND :zmax
              AND y BETWEEN :ymin AND :ymax 
              AND x BETWEEN :xmin AND :xmax;
        """
        args = self._roi_to_params(offset, shape)
        try:
            return self.cursor().execute(query, args).fetchone()[0]
        except IndexError:
            return 0

    def in_box(self, offset, shape):
        query = """
          SELECT * FROM connector 
            WHERE z BETWEEN :zmin AND :zmax
              AND y BETWEEN :ymin AND :ymax 
              AND x BETWEEN :xmin AND :xmax;
        """
        args = self._roi_to_params(offset, shape)

        for row in self.cursor().execute(query, args):
            yield row[0], CoordZYX(row[1:])

    def to_df(self):
        return pd.read_sql("SELECT * FROM connector;", self.conn)
